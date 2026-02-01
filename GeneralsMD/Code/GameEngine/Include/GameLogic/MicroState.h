/*
**	Command & Conquer Generals Zero Hour(tm)
**	Copyright 2025 Electronic Arts Inc.
**
**	This program is free software: you can redistribute it and/or modify
**	it under the terms of the GNU General Public License as published by
**	the Free Software Foundation, either version 3 of the License, or
**	(at your option) any later version.
**
**	This program is distributed in the hope that it will be useful,
**	but WITHOUT ANY WARRANTY; without even the implied warranty of
**	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
**	GNU General Public License for more details.
**
**	You should have received a copy of the GNU General Public License
**	along with this program.  If not, see <http://www.gnu.org/licenses/>.
*/

////////////////////////////////////////////////////////////////////////////////
//                                                                            //
//  Micro State - State representation for unit-level ML inference           //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

// MicroState.h
// Unit-level state for micro control decisions
// Author: Mito, 2025

#pragma once

#ifndef _MICRO_STATE_H_
#define _MICRO_STATE_H_

#include "Common/GameMemory.h"
#include "Common/GameType.h"

// Forward declarations
class Object;
class Player;

// =============================================================================
// Micro Configuration Constants
// =============================================================================
namespace MicroConfig {
    // Decision timing (in frames, 30 FPS)
    static const UnsignedInt DECISION_INTERVAL = 15;  // 0.5 seconds per unit

    // State dimensions
    static const UnsignedInt STATE_DIM = 32;

    // Combat detection
    static const Real COMBAT_DETECTION_RADIUS = 500.0f;  // Units within this are in combat
    static const Real KITE_RANGE_FACTOR = 0.8f;          // Stay at 80% of max range for kiting

    // Value thresholds
    static const Real HIGH_VALUE_COST = 1000.0f;         // Cost above this = high value unit
    static const Real CRITICAL_HEALTH = 0.25f;           // Health below this = critical
    static const Real LOW_HEALTH = 0.5f;                 // Health below this = low
}

/**
 * Micro action types corresponding to Python MicroAction enum.
 */
enum MicroActionType
{
    MICRO_ATTACK_CURRENT = 0,    // Continue current target
    MICRO_ATTACK_NEAREST = 1,    // Switch to nearest enemy
    MICRO_ATTACK_WEAKEST = 2,    // Focus weakest enemy
    MICRO_ATTACK_PRIORITY = 3,   // Attack high-value target
    MICRO_MOVE_FORWARD = 4,      // Advance toward enemy
    MICRO_MOVE_BACKWARD = 5,     // Kite (retreat while attacking)
    MICRO_MOVE_FLANK = 6,        // Circle strafe
    MICRO_HOLD_FIRE = 7,         // Stealth/hold position
    MICRO_USE_ABILITY = 8,       // Use special ability
    MICRO_RETREAT = 9,           // Full disengage
    MICRO_FOLLOW_TEAM = 10,      // Default team behavior

    MICRO_ACTION_COUNT = 11
};

/**
 * Micro state for unit-level ML inference.
 * 32 dimensions matching the Python MicroState.
 */
struct MicroState
{
    // Unit identity (4 floats)
    Real unitType;           // Encoded unit type (0-1 normalized)
    Real isHero;             // 1.0 if hero unit
    Real veterancy;          // 0-1 (0=rookie, 1=elite)
    Real hasAbility;         // 1.0 if has special ability

    // Unit status (8 floats)
    Real health;             // Current health ratio 0-1
    Real shield;             // Shield/armor bonus (if applicable)
    Real ammunition;         // Normalized ammo (1.0 = full)
    Real cooldown;           // Weapon cooldown progress (0=ready, 1=reloading)
    Real speed;              // Normalized movement speed
    Real range;              // Normalized weapon range
    Real dps;                // Normalized damage per second
    Real armor;              // Normalized armor value

    // Situational (12 floats)
    Real nearestEnemyDist;   // Distance to nearest enemy (normalized)
    Real nearestEnemyAngle;  // Angle to nearest enemy (-1 to 1, normalized from -pi to pi)
    Real nearestEnemyHealth; // Health of nearest enemy
    Real nearestEnemyThreat; // Threat level of nearest enemy
    Real nearestAllyDist;    // Distance to nearest ally
    Real inCover;            // 1.0 if in cover/garrison
    Real underFire;          // 1.0 if taking damage
    Real abilityReady;       // 1.0 if ability ready to use
    Real targetDist;         // Distance to current target
    Real targetHealth;       // Health of current target
    Real targetType;         // Type of current target (encoded)
    Real canRetreat;         // 1.0 if retreat path clear

    // Team context (4 floats)
    Real objectiveType;      // Current team objective
    Real objectiveDir;       // Direction to objective (-1 to 1)
    Real teamRole;           // Role in team (scout, dps, tank, etc.)
    Real priority;           // Priority level for micro

    // Temporal (4 floats)
    Real timeSinceHit;       // Time since last damage taken (normalized)
    Real timeSinceShot;      // Time since last shot fired (normalized)
    Real timeInCombat;       // Duration of current combat (normalized)
    Real movementHistory;    // Recent movement direction variance

    // Initialize to zeros
    void clear();

    // Serialize to float array for JSON transmission
    void toFloatArray(Real* outArray) const;

    // State dimension
    static const UnsignedInt DIM = 32;
};

/**
 * Micro command from ML system.
 */
struct MicroCommand
{
    MicroActionType action;  // Discrete action 0-10
    Real moveAngle;          // Movement angle in radians (-pi to pi)
    Real moveDistance;       // Movement distance (0-1, normalized)

    // Is this command valid?
    Bool valid;

    // Unit ID this command is for
    ObjectID unitId;

    void clear();
};

/**
 * Determine if a unit should receive micro control.
 *
 * @param unit The unit to check
 * @return true if unit should be micro-managed
 */
Bool shouldMicroUnit(Object* unit);

/**
 * Build micro state from an Object.
 *
 * @param unit The unit to extract state from
 * @param teamObjective Current team objective context
 * @param outState Output state struct
 */
void buildMicroState(
    Object* unit,
    const Real* teamObjective,
    MicroState& outState
);

/**
 * Find the nearest enemy to a unit.
 *
 * @param unit The reference unit
 * @param outDist Output distance to enemy
 * @param outAngle Output angle to enemy
 * @return Pointer to nearest enemy, or NULL if none
 */
Object* findNearestEnemy(Object* unit, Real* outDist, Real* outAngle);

/**
 * Find the weakest enemy in range.
 *
 * @param unit The reference unit
 * @param range Maximum range to search
 * @return Pointer to weakest enemy, or NULL if none
 */
Object* findWeakestEnemy(Object* unit, Real range);

/**
 * Find a high-priority target.
 *
 * @param unit The reference unit
 * @param range Maximum range to search
 * @return Pointer to priority target, or NULL if none
 */
Object* findPriorityTarget(Object* unit, Real range);

/**
 * Calculate kite position (move away while staying in range).
 *
 * @param unit The unit to kite with
 * @param targetPos Position of target to kite from
 * @param outPos Output position to move to
 */
void calculateKitePosition(Object* unit, const Coord3D* targetPos, Coord3D* outPos);

/**
 * Calculate flank position (circle strafe).
 *
 * @param unit The unit to flank with
 * @param targetPos Position to flank around
 * @param clockwise Direction to circle
 * @param outPos Output position to move to
 */
void calculateFlankPosition(Object* unit, const Coord3D* targetPos, Bool clockwise, Coord3D* outPos);

/**
 * Check if retreat path is clear.
 *
 * @param unit The unit to check retreat for
 * @param basePos Position of friendly base
 * @return 1.0 if clear, 0.0 if blocked
 */
Real checkRetreatPath(Object* unit, const Coord3D* basePos);

/**
 * Get unit type encoding.
 *
 * @param unit The unit to classify
 * @return Normalized type value 0-1
 */
Real getUnitTypeEncoding(Object* unit);

/**
 * Calculate unit's DPS (damage per second).
 *
 * @param unit The unit to analyze
 * @return Normalized DPS value
 */
Real calculateUnitDPS(Object* unit);

#endif // _MICRO_STATE_H_
