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

// MLBridge.cpp
// Named pipe communication with Python ML process
// Author: Mito, 2025

#include "PreRTS.h"

#define WIN32_LEAN_AND_MEAN
#include <windows.h>

#include "GameLogic/MLBridge.h"
#include "GameLogic/GameLogic.h"

#ifdef _INTERNAL
// for occasional debugging...
//#pragma optimize("", off)
//#pragma MESSAGE("************************************** WARNING, optimization disabled for debugging purposes")
#endif

///////////////////////////////////////////////////////////////////////////////////////////////////
// MLGameState
///////////////////////////////////////////////////////////////////////////////////////////////////

void MLGameState::clear()
{
	playerIndex = 0;
	money = 0.0f;
	powerBalance = 0.0f;
	incomeRate = 0.0f;
	supplyUsed = 0.0f;

	for (int i = 0; i < 3; i++) {
		ownInfantry[i] = 0.0f;
		ownVehicles[i] = 0.0f;
		ownAircraft[i] = 0.0f;
		ownStructures[i] = 0.0f;
		enemyInfantry[i] = 0.0f;
		enemyVehicles[i] = 0.0f;
		enemyAircraft[i] = 0.0f;
		enemyStructures[i] = 0.0f;
	}

	gameTimeMinutes = 0.0f;
	techLevel = 0.0f;
	baseThreat = 0.0f;
	armyStrength = 0.0f;
	underAttack = 0.0f;
	distanceToEnemy = 0.0f;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// MLRecommendation
///////////////////////////////////////////////////////////////////////////////////////////////////

void MLRecommendation::clear()
{
	// Default to balanced priorities
	priorityEconomy = 0.25f;
	priorityDefense = 0.25f;
	priorityMilitary = 0.25f;
	priorityTech = 0.25f;

	// Default to balanced army
	preferInfantry = 0.33f;
	preferVehicles = 0.34f;
	preferAircraft = 0.33f;

	// Moderate aggression
	aggression = 0.5f;

	// Auto-select target
	targetPlayer = -1;

	valid = false;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// MLBridge Constructor / Destructor
///////////////////////////////////////////////////////////////////////////////////////////////////

MLBridge::MLBridge() :
	m_pipeHandle(INVALID_HANDLE_VALUE),
	m_connected(false),
	m_hasRecommendation(false),
	m_lastConnectAttempt(0),
	m_trainerLaunched(false),
	m_trainerProcess(NULL)
{
	m_lastRecommendation.clear();
	memset(m_readBuffer, 0, READ_BUFFER_SIZE);
}

MLBridge::~MLBridge()
{
	disconnect();

	// Clean up trainer process handle (don't terminate - let it finish)
	if (m_trainerProcess != NULL) {
		CloseHandle((HANDLE)m_trainerProcess);
		m_trainerProcess = NULL;
	}
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Connection Management
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool MLBridge::connect()
{
	if (m_connected) {
		return true;
	}

	// Throttle reconnection attempts
	UnsignedInt currentFrame = TheGameLogic ? TheGameLogic->getFrame() : 0;
	if (currentFrame - m_lastConnectAttempt < RECONNECT_INTERVAL_FRAMES) {
		return false;
	}
	m_lastConnectAttempt = currentFrame;

	// Try to connect to existing pipe (Python server should create it)
	HANDLE pipe = CreateFileA(
		getPipeName(),
		GENERIC_READ | GENERIC_WRITE,
		0,              // No sharing
		NULL,           // Default security
		OPEN_EXISTING,  // Must already exist
		FILE_FLAG_OVERLAPPED,  // For non-blocking I/O
		NULL            // No template
	);

	if (pipe == INVALID_HANDLE_VALUE) {
		DWORD error = GetLastError();
		if (error == ERROR_FILE_NOT_FOUND) {
			// Pipe doesn't exist - try to launch the trainer
			if (!m_trainerLaunched) {
				DEBUG_LOG(("MLBridge: Pipe not found, launching trainer...\n"));
				launchTrainer();
			}
		} else if (error != ERROR_PIPE_BUSY) {
			DEBUG_LOG(("MLBridge: Failed to connect, error %d\n", error));
		}
		return false;
	}

	// Set pipe to message mode
	DWORD mode = PIPE_READMODE_BYTE;
	if (!SetNamedPipeHandleState(pipe, &mode, NULL, NULL)) {
		DEBUG_LOG(("MLBridge: Failed to set pipe mode\n"));
		CloseHandle(pipe);
		return false;
	}

	m_pipeHandle = pipe;
	m_connected = true;
	DEBUG_LOG(("MLBridge: Connected to Python ML server\n"));

	return true;
}

void MLBridge::disconnect()
{
	if (m_pipeHandle != INVALID_HANDLE_VALUE) {
		CloseHandle((HANDLE)m_pipeHandle);
		m_pipeHandle = INVALID_HANDLE_VALUE;
	}
	m_connected = false;
	m_hasRecommendation = false;
	DEBUG_LOG(("MLBridge: Disconnected\n"));
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Trainer Process Launch
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool MLBridge::launchTrainer()
{
	if (m_trainerLaunched) {
		return true;  // Already launched
	}

	m_trainerLaunched = true;  // Mark as attempted even if it fails

	// Get the game's directory
	char gameDir[MAX_PATH];
	GetModuleFileNameA(NULL, gameDir, MAX_PATH);

	// Remove executable name to get directory
	char* lastSlash = strrchr(gameDir, '\\');
	if (lastSlash) {
		*lastSlash = '\0';
	}

	// Build path to Python trainer script
	// Expected location: <game_dir>/python/train_manual.py
	char scriptPath[MAX_PATH];
	sprintf(scriptPath, "%s\\python\\train_manual.py", gameDir);

	// Check if script exists
	DWORD attrs = GetFileAttributesA(scriptPath);
	if (attrs == INVALID_FILE_ATTRIBUTES) {
		DEBUG_LOG(("MLBridge: Trainer script not found at %s\n", scriptPath));
		return false;
	}

	// Build command line
	// Use pythonw.exe for no console window, or python.exe for debugging
	char cmdLine[MAX_PATH * 2];
	sprintf(cmdLine, "pythonw.exe \"%s\" --episodes 9999", scriptPath);

	DEBUG_LOG(("MLBridge: Launching trainer: %s\n", cmdLine));

	// Set up process creation
	STARTUPINFOA si;
	PROCESS_INFORMATION pi;
	ZeroMemory(&si, sizeof(si));
	si.cb = sizeof(si);
	ZeroMemory(&pi, sizeof(pi));

	// Set working directory to game's python folder
	char workDir[MAX_PATH];
	sprintf(workDir, "%s\\python", gameDir);

	// Create the process
	BOOL success = CreateProcessA(
		NULL,           // Application name (NULL = use command line)
		cmdLine,        // Command line
		NULL,           // Process security attributes
		NULL,           // Thread security attributes
		FALSE,          // Don't inherit handles
		CREATE_NO_WINDOW | DETACHED_PROCESS,  // Creation flags
		NULL,           // Use parent's environment
		workDir,        // Working directory
		&si,            // Startup info
		&pi             // Process info output
	);

	if (!success) {
		DWORD error = GetLastError();
		(void)error;  // Suppress unused warning in release builds
		DEBUG_LOG(("MLBridge: Failed to launch trainer, error %d\n", error));

		// Try with python.exe instead (might not have pythonw)
		sprintf(cmdLine, "python.exe \"%s\" --episodes 9999", scriptPath);
		success = CreateProcessA(NULL, cmdLine, NULL, NULL, FALSE,
			CREATE_NO_WINDOW | DETACHED_PROCESS, NULL, workDir, &si, &pi);

		if (!success) {
			DEBUG_LOG(("MLBridge: Also failed with python.exe, error %d\n", GetLastError()));
			return false;
		}
	}

	// Store process handle for cleanup, close thread handle
	m_trainerProcess = pi.hProcess;
	CloseHandle(pi.hThread);

	DEBUG_LOG(("MLBridge: Trainer launched (PID %d)\n", pi.dwProcessId));

	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Message I/O
///////////////////////////////////////////////////////////////////////////////////////////////////

Bool MLBridge::writeMessage(const char* data, UnsignedInt length)
{
	if (!m_connected || m_pipeHandle == INVALID_HANDLE_VALUE) {
		return false;
	}

	// Protocol: 4-byte length prefix + data
	DWORD bytesWritten = 0;

	// Write length
	if (!WriteFile((HANDLE)m_pipeHandle, &length, sizeof(UnsignedInt), &bytesWritten, NULL)) {
		DEBUG_LOG(("MLBridge: Write length failed, disconnecting\n"));
		disconnect();
		return false;
	}

	// Write data
	if (!WriteFile((HANDLE)m_pipeHandle, data, length, &bytesWritten, NULL)) {
		DEBUG_LOG(("MLBridge: Write data failed, disconnecting\n"));
		disconnect();
		return false;
	}

	return true;
}

Bool MLBridge::readMessage(char* buffer, UnsignedInt bufferSize, UnsignedInt& outLength)
{
	if (!m_connected || m_pipeHandle == INVALID_HANDLE_VALUE) {
		return false;
	}

	// Check if data is available (non-blocking)
	DWORD bytesAvailable = 0;
	if (!PeekNamedPipe((HANDLE)m_pipeHandle, NULL, 0, NULL, &bytesAvailable, NULL)) {
		DWORD error = GetLastError();
		if (error == ERROR_BROKEN_PIPE || error == ERROR_PIPE_NOT_CONNECTED) {
			DEBUG_LOG(("MLBridge: Pipe broken, disconnecting\n"));
			disconnect();
		}
		return false;
	}

	if (bytesAvailable < sizeof(UnsignedInt)) {
		return false;  // No complete message yet
	}

	// Read length prefix
	DWORD bytesRead = 0;
	UnsignedInt msgLength = 0;
	if (!ReadFile((HANDLE)m_pipeHandle, &msgLength, sizeof(UnsignedInt), &bytesRead, NULL)) {
		DEBUG_LOG(("MLBridge: Read length failed\n"));
		disconnect();
		return false;
	}

	if (msgLength >= bufferSize) {
		DEBUG_LOG(("MLBridge: Message too large (%d bytes)\n", msgLength));
		disconnect();
		return false;
	}

	// Read message data
	if (!ReadFile((HANDLE)m_pipeHandle, buffer, msgLength, &bytesRead, NULL)) {
		DEBUG_LOG(("MLBridge: Read data failed\n"));
		disconnect();
		return false;
	}

	buffer[msgLength] = '\0';
	outLength = msgLength;
	return true;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// State Export
///////////////////////////////////////////////////////////////////////////////////////////////////

AsciiString MLBridge::stateToJson(const MLGameState& state)
{
	// Simple JSON serialization (no external library needed)
	char buffer[2048];

	sprintf(buffer,
		"{"
		"\"player\":%d,"
		"\"money\":%.2f,"
		"\"power\":%.2f,"
		"\"income\":%.2f,"
		"\"supply\":%.2f,"
		"\"own_infantry\":[%.2f,%.2f,%.2f],"
		"\"own_vehicles\":[%.2f,%.2f,%.2f],"
		"\"own_aircraft\":[%.2f,%.2f,%.2f],"
		"\"own_structures\":[%.2f,%.2f,%.2f],"
		"\"enemy_infantry\":[%.2f,%.2f,%.2f],"
		"\"enemy_vehicles\":[%.2f,%.2f,%.2f],"
		"\"enemy_aircraft\":[%.2f,%.2f,%.2f],"
		"\"enemy_structures\":[%.2f,%.2f,%.2f],"
		"\"game_time\":%.2f,"
		"\"tech_level\":%.2f,"
		"\"base_threat\":%.2f,"
		"\"army_strength\":%.2f,"
		"\"under_attack\":%.1f,"
		"\"distance_to_enemy\":%.2f"
		"}",
		state.playerIndex,
		state.money,
		state.powerBalance,
		state.incomeRate,
		state.supplyUsed,
		state.ownInfantry[0], state.ownInfantry[1], state.ownInfantry[2],
		state.ownVehicles[0], state.ownVehicles[1], state.ownVehicles[2],
		state.ownAircraft[0], state.ownAircraft[1], state.ownAircraft[2],
		state.ownStructures[0], state.ownStructures[1], state.ownStructures[2],
		state.enemyInfantry[0], state.enemyInfantry[1], state.enemyInfantry[2],
		state.enemyVehicles[0], state.enemyVehicles[1], state.enemyVehicles[2],
		state.enemyAircraft[0], state.enemyAircraft[1], state.enemyAircraft[2],
		state.enemyStructures[0], state.enemyStructures[1], state.enemyStructures[2],
		state.gameTimeMinutes,
		state.techLevel,
		state.baseThreat,
		state.armyStrength,
		state.underAttack,
		state.distanceToEnemy
	);

	return AsciiString(buffer);
}

Bool MLBridge::sendState(const MLGameState& state)
{
	if (!m_connected) {
		// Try to connect
		if (!connect()) {
			return false;
		}
	}

	AsciiString json = stateToJson(state);
	return writeMessage(json.str(), json.getLength());
}

Bool MLBridge::sendGameEnd(Bool victory, Real gameTimeMinutes, Real finalArmyStrength)
{
	if (!m_connected) {
		return false;
	}

	// Send game-end notification as JSON
	char buffer[256];
	sprintf(buffer,
		"{"
		"\"type\":\"game_end\","
		"\"victory\":%s,"
		"\"game_time\":%.2f,"
		"\"army_strength\":%.2f"
		"}",
		victory ? "true" : "false",
		gameTimeMinutes,
		finalArmyStrength
	);

	Bool result = writeMessage(buffer, strlen(buffer));
	DEBUG_LOG(("MLBridge: Sent game end notification (victory=%d)\n", victory));
	return result;
}

///////////////////////////////////////////////////////////////////////////////////////////////////
// Recommendation Parsing
///////////////////////////////////////////////////////////////////////////////////////////////////

// Simple JSON number parser (finds "key":value and returns value)
static Real parseJsonFloat(const char* json, const char* key, Real defaultValue)
{
	char searchKey[64];
	sprintf(searchKey, "\"%s\":", key);

	const char* pos = strstr(json, searchKey);
	if (!pos) return defaultValue;

	pos += strlen(searchKey);
	while (*pos == ' ') pos++;

	return (Real)atof(pos);
}

static Int parseJsonInt(const char* json, const char* key, Int defaultValue)
{
	char searchKey[64];
	sprintf(searchKey, "\"%s\":", key);

	const char* pos = strstr(json, searchKey);
	if (!pos) return defaultValue;

	pos += strlen(searchKey);
	while (*pos == ' ') pos++;

	return atoi(pos);
}

Bool MLBridge::parseRecommendation(const char* json, MLRecommendation& outRec)
{
	outRec.clear();

	// Parse build priorities
	outRec.priorityEconomy = parseJsonFloat(json, "priority_economy", 0.25f);
	outRec.priorityDefense = parseJsonFloat(json, "priority_defense", 0.25f);
	outRec.priorityMilitary = parseJsonFloat(json, "priority_military", 0.25f);
	outRec.priorityTech = parseJsonFloat(json, "priority_tech", 0.25f);

	// Parse army preferences
	outRec.preferInfantry = parseJsonFloat(json, "prefer_infantry", 0.33f);
	outRec.preferVehicles = parseJsonFloat(json, "prefer_vehicles", 0.34f);
	outRec.preferAircraft = parseJsonFloat(json, "prefer_aircraft", 0.33f);

	// Parse aggression
	outRec.aggression = parseJsonFloat(json, "aggression", 0.5f);

	// Parse target
	outRec.targetPlayer = parseJsonInt(json, "target_player", -1);

	outRec.valid = true;
	return true;
}

Bool MLBridge::receiveRecommendation(MLRecommendation& outRec)
{
	if (!m_connected) {
		return false;
	}

	UnsignedInt length = 0;
	if (!readMessage(m_readBuffer, READ_BUFFER_SIZE - 1, length)) {
		return false;
	}

	if (parseRecommendation(m_readBuffer, outRec)) {
		m_lastRecommendation = outRec;
		m_hasRecommendation = true;
		return true;
	}

	return false;
}
