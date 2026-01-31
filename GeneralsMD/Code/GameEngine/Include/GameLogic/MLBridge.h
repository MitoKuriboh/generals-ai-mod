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
//  ML Bridge - Communication interface between game and Python ML system     //
//                                                                            //
////////////////////////////////////////////////////////////////////////////////

// MLBridge.h
// Named pipe communication with Python ML process
// Author: Mito, 2025

#pragma once

#ifndef _ML_BRIDGE_H_
#define _ML_BRIDGE_H_

#include "Common/GameMemory.h"
#include "Common/AsciiString.h"

// Forward declarations
class Player;
class Object;

/**
 * Game state exported to ML system.
 * All values normalized to useful ranges for neural network input.
 */
struct MLGameState
{
	// Player ID for this state
	Int playerIndex;

	// Economy (4 floats)
	Real money;              // Current money (log-scaled)
	Real powerBalance;       // Power production - consumption
	Real incomeRate;         // Approximate income per second
	Real supplyUsed;         // Supply capacity used ratio (0-1)

	// Own forces by category (count, health ratio, in production)
	Real ownInfantry[3];
	Real ownVehicles[3];
	Real ownAircraft[3];
	Real ownStructures[3];

	// Visible enemy forces (same format)
	Real enemyInfantry[3];
	Real enemyVehicles[3];
	Real enemyAircraft[3];
	Real enemyStructures[3];

	// Strategic situation
	Real gameTimeMinutes;    // Game time in minutes
	Real techLevel;          // 0-1 based on unlocked buildings
	Real baseThreat;         // Enemy units near base (0-1)
	Real armyStrength;       // Relative army value vs enemy
	Real underAttack;        // 1.0 if base under attack, else 0
	Real distanceToEnemy;    // Normalized distance to nearest enemy base

	// Initialize to zeros
	void clear();
};

/**
 * ML recommendations for AI decisions.
 * Values are preferences/weights, not absolute commands.
 */
struct MLRecommendation
{
	// Build priority weights (should sum to ~1.0)
	Real priorityEconomy;    // Supply depots, power plants
	Real priorityDefense;    // Turrets, defensive structures
	Real priorityMilitary;   // Combat units
	Real priorityTech;       // Tech buildings, upgrades

	// Army composition preference
	Real preferInfantry;
	Real preferVehicles;
	Real preferAircraft;

	// Aggression level (0 = defensive, 1 = aggressive)
	Real aggression;

	// Target player index (-1 = auto-select)
	Int targetPlayer;

	// Is this recommendation valid?
	Bool valid;

	// Initialize with defaults
	void clear();
};

/**
 * ML Bridge - Handles communication with Python ML process via named pipe.
 *
 * Protocol:
 * - Game sends: length (4 bytes) + JSON state
 * - Python sends: length (4 bytes) + JSON recommendation
 * - Non-blocking reads to avoid stalling game
 */
class MLBridge
{
public:
	MLBridge();
	~MLBridge();

	// Connection management
	Bool connect();
	void disconnect();
	Bool isConnected() const { return m_connected; }

	// Launch the Python trainer process
	Bool launchTrainer();

	// State export (call every ML_DECISION_INTERVAL frames)
	Bool sendState(const MLGameState& state);

	// Send game-end notification (victory/defeat)
	Bool sendGameEnd(Bool victory, Real gameTimeMinutes, Real finalArmyStrength);

	// Recommendation retrieval (non-blocking)
	Bool receiveRecommendation(MLRecommendation& outRec);

	// Check if we have a pending recommendation
	Bool hasRecommendation() const { return m_hasRecommendation; }

	// Get last received recommendation
	const MLRecommendation& getLastRecommendation() const { return m_lastRecommendation; }

	// Pipe name for connection
	static const char* getPipeName() { return "\\\\.\\pipe\\generals_ml_bridge"; }

private:
	// Serialize state to JSON
	AsciiString stateToJson(const MLGameState& state);

	// Parse recommendation from JSON
	Bool parseRecommendation(const char* json, MLRecommendation& outRec);

	// Low-level pipe operations
	Bool writeMessage(const char* data, UnsignedInt length);
	Bool readMessage(char* buffer, UnsignedInt bufferSize, UnsignedInt& outLength);

	// Connection state
	void* m_pipeHandle;  // HANDLE, but avoid Windows.h in header
	Bool m_connected;

	// Recommendation state
	Bool m_hasRecommendation;
	MLRecommendation m_lastRecommendation;

	// Read buffer
	static const UnsignedInt READ_BUFFER_SIZE = 4096;
	char m_readBuffer[READ_BUFFER_SIZE];

	// Reconnection throttle
	UnsignedInt m_lastConnectAttempt;
	static const UnsignedInt RECONNECT_INTERVAL_FRAMES = 300; // ~10 seconds

	// Trainer process management
	Bool m_trainerLaunched;
	void* m_trainerProcess;  // HANDLE to trainer process
};

#endif // _ML_BRIDGE_H_
