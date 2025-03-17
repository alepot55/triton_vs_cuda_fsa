#pragma once

#include <vector>
#include <map>
#include <string>
#include <set>

// Helper structure for NFA states
struct NFAState {
    std::map<char, std::set<int>> transitions; // Map from symbol to set of target states
    std::map<char, std::set<int>> epsilon_transitions; // Epsilon transitions
};

// NFA structure for regex conversion
struct NFA {
    std::vector<NFAState> states;
    int start_state;
    std::set<int> accepting_states;
};

// Helper functions
std::set<int> epsilonClosure(const NFA& nfa, const std::set<int>& states);
std::set<int> move(const NFA& nfa, const std::set<int>& states, char symbol);
