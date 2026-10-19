// Copyright © 2020-2025 Ryabkov Oleg Igorevich, Evstigneev Nikolay Mikhaylovitch

// This file is part of NMFD.

// NMFD is free software: you can redistribute it and/or modify
// it under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 2 only of the License.

// NMFD is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// You should have received a copy of the GNU General Public License
// along with NMFD.  If not, see <http://www.gnu.org/licenses/>.

#ifndef __NMFD_PRECONDITIONER_MG_H__
#define __NMFD_PRECONDITIONER_MG_H__

#include <vector>
#include <memory>
#include <stdexcept>
#include <type_traits>
#include <utility>
#include <scfd/utils/logged_obj_base.h>
#ifdef NMFD_ENABLE_NLOHMANN
#    include <nlohmann/json.hpp>
#endif
#include <nmfd/detail/vector_wrap.h>
#include <nmfd/detail/operator_update.h>
//#include <glued_matrix_operator.h>
#include "preconditioner_interface.h"
#include "mg_observer.h"

namespace nmfd
{
namespace preconditioners
{

namespace mg_detail
{

template <class Coarsening, class Operator, class = void>
struct has_same_structure : std::false_type
{
};

template <class Coarsening, class Operator>
struct has_same_structure<
    Coarsening, Operator,
    std::void_t<decltype( std::declval<const Coarsening &>()
                              .same_structure( std::declval<const Operator &>(), std::declval<const Operator &>() ) )>>
    : std::true_type
{
};

template <class Coarsening, class Operator>
bool same_structure( const Coarsening &coarsening, const Operator &current, const Operator &updated )
{
    if constexpr ( has_same_structure<Coarsening, Operator>::value )
        return coarsening.same_structure( current, updated );
    else
        return true;
}

template <class Coarsening, class Operator, class Restrictor, class Prolongator, class = void>
struct has_update_coarse_operator : std::false_type
{
};

template <class Coarsening, class Operator, class Restrictor, class Prolongator>
struct has_update_coarse_operator<
    Coarsening, Operator, Restrictor, Prolongator,
    std::void_t<decltype( std::declval<Coarsening &>().update_coarse_operator(
        std::declval<const Operator &>(), std::declval<const Restrictor &>(), std::declval<Prolongator &>(),
        std::declval<std::shared_ptr<const Operator>>()
    ) )>> : std::true_type
{
};

template <class Coarsening, class Operator, class Restrictor, class Prolongator>
std::shared_ptr<const Operator> update_coarse_operator(
    Coarsening &coarsening, const Operator &fine_operator, const Restrictor &restrictor, Prolongator &prolongator,
    std::shared_ptr<const Operator> current_coarse_operator
)
{
    if constexpr ( has_update_coarse_operator<Coarsening, Operator, Restrictor, Prolongator>::value )
    {
        return coarsening.update_coarse_operator(
            fine_operator, restrictor, prolongator, std::move( current_coarse_operator )
        );
    }
    else
    {
        return coarsening.coarse_operator( fine_operator, restrictor, prolongator );
    }
}

} // namespace mg_detail

/// SystemOperator is OperatorWithSpaces.
/// Restrictor, Prolongator are Operator.
/// Smoother, CoarseSolver are Preconditioner
/// Coarsening is Coarsening
/// Smoother, CoarseSolver, Coarsening are HierarchicAlgorithm
/// All vector_space_type, vector_type, scalar_type are the same
template <
    class SystemOperator, class Restrictor, class Prolongator, class Smoother, class CoarseSolver, class Coarsening,
    class Log, class Observer = null_mg_observer>
class mg : public preconditioner_interface<typename SystemOperator::vector_space_type, SystemOperator>,
           public scfd::utils::logged_obj_base<Log>
{
public:
    using vector_space_type  = typename SystemOperator::vector_space_type;
    using vector_type        = typename vector_space_type::vector_type;
    using scalar_type        = typename vector_space_type::scalar_type;
    using operator_type      = SystemOperator;
    using restrictor_type    = Restrictor;
    using prolongator_type   = Prolongator;
    using smoother_type      = Smoother;
    using coarse_solver_type = CoarseSolver;
    using coarsening_type    = Coarsening;
    using observer_type      = Observer;

    using T                   = scalar_type;
    using logged_obj_t        = scfd::utils::logged_obj_base<Log>;
    using logged_obj_params_t = typename logged_obj_t::params;

    struct params : public logged_obj_params_t
    {
        std::size_t max_levels, cycle_type, num_sweeps_pre, num_sweeps_post;
        bool        direct_coarse;
        //TODO not implemented
        std::string out_prefix;
        bool        set_direct_coarse_matrix_defect;
        bool        regularize_after_direct_coarse;

        params( const std::string &log_prefix = "", const std::string &log_name = "mg::" )
            : logged_obj_params_t( 0, log_prefix + log_name ), max_levels( 25 ), cycle_type( 1 ), num_sweeps_pre( 1 ),
              num_sweeps_post( 1 ), direct_coarse( true ), out_prefix( "mg_" ),
              set_direct_coarse_matrix_defect( false ), regularize_after_direct_coarse( false )
        {
        }
#ifdef NMFD_ENABLE_NLOHMANN
        void from_json( const nlohmann::json &j )
        {
        }
        nlohmann::json to_json() const
        {
            return nlohmann::json();
        }
#endif
    };
    using smoother_params_hierarchy_type      = typename nmfd::detail::algo_params_hierarchy<smoother_type>::type;
    using coarse_solver_params_hierarchy_type = typename nmfd::detail::algo_params_hierarchy<coarse_solver_type>::type;
    using coarsening_params_hierarchy_type    = typename nmfd::detail::algo_params_hierarchy<coarsening_type>::type;
    struct params_hierarchy : public params
    {
        smoother_params_hierarchy_type      smoother;
        coarse_solver_params_hierarchy_type coarse_solver;
        coarsening_params_hierarchy_type    coarsening;

        //TODO add prefix for other subalgorithms
        params_hierarchy( const std::string &log_prefix = "", const std::string &log_name = "mg::" )
            : params( log_prefix, log_name ), smoother( this->log_msg_prefix )
        {
        }
        params_hierarchy(
            const params &prm_, const smoother_params_hierarchy_type &smoother_,
            const coarse_solver_params_hierarchy_type &coarse_solver_,
            const coarsening_params_hierarchy_type    &coarsening_
        )
            : params( prm_ ), smoother( smoother_ ), coarse_solver( coarse_solver_ ), coarsening( coarsening_ )
        {
        }
#ifdef NMFD_ENABLE_NLOHMANN
        void from_json( const nlohmann::json &j )
        {
            params::from_json( j );
            smoother.from_json( j.at( "smoother" ) );
            coarse_solver.from_json( j.at( "coarse_solver" ) );
            coarsening.from_json( j.at( "coarsening" ) );
        }
        nlohmann::json to_json() const
        {
            nlohmann::json j = params::to_json(), j_smoother = smoother.to_json(),
                           j_coarse_solver = coarse_solver.to_json(), j_coarsening = coarsening.to_json();
            j["smoother"]      = j_smoother;
            j["coarse_solver"] = j_coarse_solver;
            j["coarsening"]    = j_coarsening;
            return j;
        }
#endif
    };
    using smoother_utils_hierarchy_type      = typename nmfd::detail::algo_utils_hierarchy<smoother_type>::type;
    using coarse_solver_utils_hierarchy_type = typename nmfd::detail::algo_utils_hierarchy<coarse_solver_type>::type;
    using coarsening_utils_hierarchy_type    = typename nmfd::detail::algo_utils_hierarchy<coarsening_type>::type;
    struct utils
    {
        Log          *log;
        observer_type observer;
        utils( Log *log_ = nullptr, observer_type observer_ = observer_type() )
            : log( log_ ), observer( std::move( observer_ ) )
        {
        }
    };
    struct utils_hierarchy : public utils
    {
        smoother_utils_hierarchy_type      smoother;
        coarse_solver_utils_hierarchy_type coarse_solver;
        coarsening_utils_hierarchy_type    coarsening;

        utils_hierarchy() = default;
        utils_hierarchy( Log *log_, observer_type observer_ = observer_type() ) : utils( log_, std::move( observer_ ) )
        {
        }
    };

    struct status
    {
        std::size_t levels           = 0;
        std::size_t apply_calls      = 0;
        std::size_t operator_updates = 0;
    };

    mg( const utils_hierarchy &u, const params_hierarchy &p ) : logged_obj_t( u.log, p ), utils_( u ), prm_( p )
    {
    }
    ~mg()
    {
    }

    // Rebuilds hierarchy structure, including transfer operators.
    void set_operator( std::shared_ptr<const operator_type> op ) override
    {
        build( op );
    }

    // Preserves hierarchy topology and refreshes only numerical operator data.
    // The caller must provide an operator with the same discrete structure.
    void update_operator( std::shared_ptr<const operator_type> op ) override
    {
        update( std::move( op ) );
    }

    bool is_ready() const noexcept
    {
        return !levs_.empty();
    }

    std::size_t level_count() const noexcept
    {
        return levs_.size();
    }

    status get_status() const noexcept
    {
        return status{ levs_.size(), apply_calls_, operator_updates_ };
    }

    void apply( const vector_type &rhs, vector_type &x ) const
    {
        if ( levs_.empty() )
            throw std::logic_error( "mg::apply: levels are empty" );

        ++apply_calls_;
        levs_[0].vec_sp->assign( rhs, *levs_[0].rhs );
        cycle( 0 );
        levs_[0].vec_sp->assign( *levs_[0].x, x );
    }

    /// inplace version for preconditioner interface
    void apply( vector_type &x ) const
    {
        if ( levs_.empty() )
            throw std::logic_error( "mg::apply: levels are empty" );

        ++apply_calls_;
        levs_[0].vec_sp->assign( x, *levs_[0].rhs );
        cycle( 0 );
        levs_[0].vec_sp->assign( *levs_[0].x, x );
    }

private:
    //TODO make it true,false and add start_use helper class in apply
    using buf_arr_t = detail::vector_wrap<vector_space_type, true, true>;
    struct level_t
    {
        std::shared_ptr<const operator_type> sys_operator;
        std::shared_ptr<restrictor_type>     restrictor;
        std::shared_ptr<prolongator_type>    prolongator;

        std::shared_ptr<smoother_type>      smoother;
        std::shared_ptr<coarse_solver_type> coarse_solver;

        std::shared_ptr<vector_space_type> vec_sp;
        buf_arr_t                          x, residual, rhs;

        level_t(
            std::shared_ptr<const operator_type> op, const utils_hierarchy &utils, const params_hierarchy &prm,
            bool create_coarse_solver = false
        )
            : sys_operator( std::move( op ) ), vec_sp( sys_operator->get_dom_space() ), x( *vec_sp ),
              residual( *vec_sp ), rhs( *vec_sp )
        {
            smoother = algo_hierarchy_creator<smoother_type>::get( utils.smoother, prm.smoother );
            smoother->set_operator( sys_operator );
            if ( create_coarse_solver )
            {
                coarse_solver =
                    algo_hierarchy_creator<coarse_solver_type>::get( utils.coarse_solver, prm.coarse_solver );
                coarse_solver->set_operator( sys_operator );
            }
        }
        level_t( const level_t & )            = delete;
        level_t &operator=( const level_t & ) = delete;
        level_t( level_t && )                 = default;
        level_t &operator=( level_t && )      = default;

        std::shared_ptr<operator_type> create_next( coarsening_type &c )
        {
            auto transfer_ops = c.next_level( *sys_operator );
            restrictor        = std::get<0>( transfer_ops );
            prolongator       = std::get<1>( transfer_ops );
            if ( restrictor )
                return c.coarse_operator( *sys_operator, *restrictor, *prolongator );
            else
                return std::shared_ptr<operator_type>();
        }

        void create_coarse_solver( const utils_hierarchy &utils, const params_hierarchy &prm )
        {
            if ( coarse_solver )
                return;
            coarse_solver = algo_hierarchy_creator<coarse_solver_type>::get( utils.coarse_solver, prm.coarse_solver );
            coarse_solver->set_operator( sys_operator );
        }

        void update_operator( std::shared_ptr<const operator_type> op )
        {
            if ( !op )
                throw std::invalid_argument( "mg::level_t::update_operator: operator is null" );
            nmfd::detail::update_operator_or_set( *smoother, op );
            if ( coarse_solver )
                nmfd::detail::update_operator_or_set( *coarse_solver, op );
            sys_operator = std::move( op );
        }
        /// Calcs residual using x
        void calc_residual()
        {
            sys_operator->apply( *x, *residual );
            vec_sp->add_lin_comb( T( 1 ), *rhs, -T( 1 ), *residual );
        }
        /// Calcs next x using previous x value and precalced residual
        void make_iter()
        {
            smoother->apply( *residual );
            vec_sp->add_lin_comb( T( 1 ), *residual, T( 1 ), *x );
        }
    };

    utils_hierarchy                  utils_;
    params_hierarchy                 prm_;
    std::shared_ptr<coarsening_type> coarsening_;
    /// TODO mutable - because of rhs x residual?
    mutable std::vector<level_t> levs_;
    mutable std::size_t          apply_calls_      = 0;
    std::size_t                  operator_updates_ = 0;

    void build( std::shared_ptr<const operator_type> op )
    {
        if ( !op )
            throw std::invalid_argument( "mg::build: system operator is null" );
        if ( prm_.max_levels == 0 )
            throw std::invalid_argument( "mg::build: max_levels must be positive" );
        if ( prm_.cycle_type == 0 )
            throw std::invalid_argument( "mg::build: cycle_type must be positive" );

        std::vector<level_t> candidate;
        auto candidate_coarsening = algo_hierarchy_creator<coarsening_type>::get( utils_.coarsening, prm_.coarsening );

        auto curr_op = op;
        while ( true )
        {
            const bool terminal =
                candidate_coarsening->coarse_enough( *curr_op ) || candidate.size() + 1 >= prm_.max_levels;
            candidate.emplace_back( curr_op, utils_, prm_, terminal && prm_.direct_coarse );
            if ( terminal )
                break;

            if ( prm_.out_prefix != "" )
            {
                //TODO output to file
            }
            curr_op = candidate.back().create_next( *candidate_coarsening );
            if ( !curr_op )
            {
                if ( prm_.direct_coarse )
                    candidate.back().create_coarse_solver( utils_, prm_ );
                break;
            }
        }

        levs_.swap( candidate );
        coarsening_       = std::move( candidate_coarsening );
        apply_calls_      = 0;
        operator_updates_ = 0;
        logged_obj_t::info_f( "build complete: levels number = %d", levs_.size() );
    }

    void update( std::shared_ptr<const operator_type> op )
    {
        if ( !op )
            throw std::invalid_argument( "mg::update_operator: system operator is null" );
        if ( levs_.empty() )
            throw std::logic_error( "mg::update_operator: hierarchy is not built; call set_operator first" );

        if ( !coarsening_ )
            throw std::logic_error( "mg::update_operator: coarsening policy is not initialized" );
        if ( !mg_detail::same_structure( *coarsening_, *levs_.front().sys_operator, *op ) )
            throw std::invalid_argument( "mg::update_operator: operator structure differs from the active hierarchy" );

        std::vector<std::shared_ptr<const operator_type>> updated_operators;
        updated_operators.reserve( levs_.size() );
        updated_operators.emplace_back( std::move( op ) );

        for ( std::size_t level = 0; level + 1 < levs_.size(); ++level )
        {
            if ( !levs_[level].restrictor || !levs_[level].prolongator )
                throw std::logic_error( "mg::update_operator: a nonterminal level has no transfer operators" );
            auto coarse = mg_detail::update_coarse_operator(
                *coarsening_, *updated_operators.back(), *levs_[level].restrictor, *levs_[level].prolongator,
                levs_[level + 1].sys_operator
            );
            if ( !coarse )
                throw std::runtime_error( "mg::update_operator: coarse operator update returned null" );
            if ( !mg_detail::same_structure( *coarsening_, *levs_[level + 1].sys_operator, *coarse ) )
                throw std::invalid_argument( "mg::update_operator: coarse operator structure changed" );
            updated_operators.emplace_back( std::move( coarse ) );
        }

        for ( std::size_t level = 0; level < levs_.size(); ++level )
            levs_[level].update_operator( std::move( updated_operators[level] ) );

        apply_calls_ = 0;
        ++operator_updates_;
        logged_obj_t::info_f( "operator values updated: levels number = %d", levs_.size() );
    }

    void notify( std::size_t level, mg_cycle_event event, const level_t &data, const vector_type &vector ) const
    {
        utils_.observer.on_event( level, event, *data.vec_sp, vector );
    }

    void cycle( size_t levi ) const
    {
        auto &curr = levs_[levi];

        curr.vec_sp->assign_scalar( T( 0 ), *curr.x );
        curr.vec_sp->assign( *curr.rhs, *curr.residual );
        notify( levi, mg_cycle_event::cycle_begin_rhs, curr, *curr.rhs );
        notify( levi, mg_cycle_event::initial_residual, curr, *curr.residual );

        if ( levi + 1 == levs_.size() )
        {
            if ( curr.coarse_solver )
            {
                curr.coarse_solver->apply( *curr.residual, *curr.x );
                notify( levi, mg_cycle_event::coarse_solution, curr, *curr.x );
                /*if (lvl->level_regularization) {
                    lvl->level_regularization->apply(x);
                }*/
            }
            else
            {
                for ( size_t i = 0; i < prm_.num_sweeps_pre; ++i )
                {
                    curr.make_iter();
                    curr.calc_residual();
                    notify( levi, mg_cycle_event::pre_smoothing_residual, curr, *curr.residual );
                }
                for ( size_t i = 0; i < prm_.num_sweeps_post; ++i )
                {
                    curr.make_iter();
                    ///TODO can remove last call but not very important for the last level
                    curr.calc_residual();
                    notify( levi, mg_cycle_event::post_smoothing_residual, curr, *curr.residual );
                }
            }
        }
        else
        {
            auto &next = levs_[levi + 1];

            for ( size_t j = 0; j < prm_.cycle_type; ++j )
            {
                if ( j > 0 )
                {
                    curr.calc_residual();
                }

                for ( size_t i = 0; i < prm_.num_sweeps_pre; ++i )
                {
                    curr.make_iter();
                    curr.calc_residual();
                    notify( levi, mg_cycle_event::pre_smoothing_residual, curr, *curr.residual );
                    /*auto res_norm = curr.vec_sp->norm(*curr.residual);
                    logged_obj_t::info_f("cycle: level number = %d, pre_cycle res_norm = %e", levi, res_norm);*/
                }

                curr.restrictor->apply( *curr.residual, *next.rhs );
                notify( levi + 1, mg_cycle_event::restricted_rhs, next, *next.rhs );

                cycle( levi + 1 );

                /// NOTE *curr.residual is used as tmp buffer here to not create extra buffers
                curr.prolongator->apply( *next.x, *curr.residual );
                notify( levi, mg_cycle_event::prolongated_correction, curr, *curr.residual );
                curr.vec_sp->add_lin_comb( T( 1 ), *curr.residual, T( 1 ), *curr.x );

                for ( size_t i = 0; i < prm_.num_sweeps_post; ++i )
                {
                    curr.calc_residual();
                    notify( levi, mg_cycle_event::post_smoothing_residual, curr, *curr.residual );
                    /*auto res_norm = curr.vec_sp->norm(*curr.residual);
                    logged_obj_t::info_f("cycle: level number = %d, post_cycle res_norm = %e", levi, res_norm);*/
                    curr.make_iter();
                }
            }
        }
        notify( levi, mg_cycle_event::cycle_end_solution, curr, *curr.x );
    }
};


} // preconditioners
} // nmfd

#endif
