#include "fsa_engine.h"
#include <stack>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <unordered_set>

// ============ Implementazione della conversione da Regex a NFA ============

// Struttura per NFA durante la costruzione
struct NFAState {
    int id;
    std::map<char, std::vector<int>> transitions;
    std::map<int, std::vector<int>> epsilon_transitions;
    bool is_accepting;
};

// Struttura per rappresentare NFA durante la costruzione
struct NFA {
    std::vector<NFAState> states;
    int start_state;
    std::set<int> accepting_states;
    
    // Aggiunge un nuovo stato e ne ritorna l'ID
    int addState(bool accepting = false) {
        int id = states.size();
        states.push_back({id, {}, {}, accepting});
        if (accepting) {
            accepting_states.insert(id);
        }
        return id;
    }
    
    // Aggiunge una transizione epsilon
    void addEpsilonTransition(int from, int to) {
        states[from].epsilon_transitions[0].push_back(to);
    }
    
    // Aggiunge una transizione su simbolo
    void addTransition(int from, char symbol, int to) {
        states[from].transitions[symbol].push_back(to);
    }

    // Copy other states into this NFA
    void copyStatesFrom(const NFA& other, int& stateOffset) {
        for (const auto& state : other.states) {
            NFAState newState;
            newState.id = state.id + stateOffset;
            
            // Copy symbol transitions
            for (const auto& [symbol, targets] : state.transitions) {
                for (int target : targets) {
                    newState.transitions[symbol].push_back(target + stateOffset);
                }
            }
            
            // Copy epsilon transitions
            for (const auto& [eps, targets] : state.epsilon_transitions) {
                for (int target : targets) {
                    newState.epsilon_transitions[eps].push_back(target + stateOffset);
                }
            }
            
            newState.is_accepting = state.is_accepting;
            states.push_back(newState);
        }
    }
};

// Utility functions for regex parsing
bool isOperator(char c) {
    return (c == '|' || c == '.' || c == '*' || c == '(' || c == ')' || c == '?');
}

int getPrecedence(char op) {
    if (op == '|') return 1;
    if (op == '.') return 2;  // explicit concatenation
    if (op == '*' || op == '?') return 3;
    return 0;
}

#include <cctype>  // for isalnum
std::string addExplicitConcatenation(const std::string& regex) {
    std::string result;
    for (size_t i = 0; i < regex.size(); i++) {
        char c = regex[i];
        result.push_back(c);
        if (i + 1 < regex.size()) {
            char next = regex[i + 1];
            // Define an operand as an alphanumeric character, a closing parenthesis, or a quantifier.
            bool currIsOperand = (isalnum(c) || c == ')' || c == '*' || c == '?');
            // An operand for the next token is an alphanumeric character or an opening parenthesis.
            bool nextIsOperand = (isalnum(next) || next == '(');
            if (currIsOperand && nextIsOperand) {
                result.push_back('.');
            }
        }
    }
    return result;
}

// NEW: Expand simple {n} operators in regex (supports single-character or parenthesized group operands)
std::string expandRepetition(const std::string &regex) {
    std::string result;
    for (size_t i = 0; i < regex.size(); i++) {
        if (regex[i] == '{') {
            // Find matching '}'
            size_t close = regex.find('}', i);
            if (close == std::string::npos) {
                throw std::runtime_error("Mismatched curly braces in regex");
            }
            std::string numStr = regex.substr(i+1, close - i - 1);
            int count = std::stoi(numStr);
            // Determine the operand preceding '{'
            std::string operand;
            if (!result.empty() && result.back() == ')') {
                // Find matching '(' in result (assume well-formed)
                int paren = 0;
                size_t pos = result.size()-1;
                for (; pos < result.size(); pos--) {
                    if (result[pos] == ')')
                        paren++;
                    else if (result[pos] == '(') {
                        paren--;
                        if (paren == 0)
                            break;
                    }
                }
                if (pos == std::string::npos) {
                    throw std::runtime_error("Mismatched parentheses for repetition");
                }
                operand = result.substr(pos);
                result.erase(pos);
            } else {
                // Use the last character as operand
                if (!result.empty()) {
                    operand = result.substr(result.size()-1);
                    result.erase(result.size()-1);
                } else {
                    throw std::runtime_error("No operand for repetition operator");
                }
            }
            // Append operand count times (already one copy exists; here we add count copies)
            for (int j = 0; j < count; j++) {
                result += operand;
            }
            i = close; // Skip over the {...} operator
        } else {
            result.push_back(regex[i]);
        }
    }
    return result;
}

// Thompson's Construction Algorithm for regex to NFA
NFA createBasicNFA(char symbol) {
    NFA nfa;
    int start = nfa.addState();
    int end = nfa.addState(true);
    nfa.start_state = start;
    nfa.addTransition(start, symbol, end);
    return nfa;
}

NFA concatenateNFAs(const NFA& first, const NFA& second) {
    NFA result;
    
    // Add all states from first NFA
    int firstOffset = 0;
    result.copyStatesFrom(first, firstOffset);
    result.start_state = first.start_state;
    
    // Add all states from second NFA
    int secondOffset = first.states.size();
    result.copyStatesFrom(second, secondOffset);
    
    // Connect first accepting states to second start state
    for (int acceptingState : first.accepting_states) {
        result.addEpsilonTransition(acceptingState, second.start_state + secondOffset);
    }
    
    // Only states from second NFA are accepting now
    result.accepting_states.clear();
    for (int acceptingState : second.accepting_states) {
        result.accepting_states.insert(acceptingState + secondOffset);
    }
    
    return result;
}

NFA alternateNFAs(const NFA& first, const NFA& second) {
    NFA result;
    
    // Create boundary states to ensure proper pattern isolation
    int start = result.addState();
    int accept = result.addState(true);
    result.start_state = start;
    
    // Add states from both NFAs with offset
    int firstOffset = 2;
    result.copyStatesFrom(first, firstOffset);
    
    int secondOffset = firstOffset + first.states.size();
    result.copyStatesFrom(second, secondOffset);
    
    // Connect start state to both NFAs
    result.addEpsilonTransition(start, first.start_state + firstOffset);
    result.addEpsilonTransition(start, second.start_state + secondOffset);
    
    // Connect accepting states to final accept state
    for (int acceptingState : first.accepting_states) {
        result.addEpsilonTransition(acceptingState + firstOffset, accept);
    }
    for (int acceptingState : second.accepting_states) {
        result.addEpsilonTransition(acceptingState + secondOffset, accept);
    }
    
    return result;
}

NFA applyKleeneStar(const NFA& nfa) {
    NFA result;
    
    // Create special structure to handle repetitions correctly
    int start = result.addState();
    int accept = result.addState(true);
    result.start_state = start;
    
    // Add original NFA states
    int nfaOffset = 2;
    result.copyStatesFrom(nfa, nfaOffset);
    
    // Add epsilon from start to accept (match empty string)
    result.addEpsilonTransition(start, accept);
    
    // Add epsilon from start to NFA start (enter the repetition)
    result.addEpsilonTransition(start, nfa.start_state + nfaOffset);
    
    // For each accepting state in original NFA
    for (int acceptingState : nfa.accepting_states) {
        // Connect to final accept state
        result.addEpsilonTransition(acceptingState + nfaOffset, accept);
        
        // Connect back to start of pattern for repetition
        result.addEpsilonTransition(acceptingState + nfaOffset, nfa.start_state + nfaOffset);
    }
    
    return result;
}

// Add support for the optional operator (zero or one occurrence)
NFA applyOptional(const NFA& nfa) {
    NFA result;
    int start = result.addState();
    int accept = result.addState(true);
    result.start_state = start;
    int nfaOffset = 2;
    result.copyStatesFrom(nfa, nfaOffset);
    // Optional: allow skipping the operand
    result.addEpsilonTransition(start, accept);
    // Also allow entering the operand and finishing after it
    result.addEpsilonTransition(start, nfa.start_state + nfaOffset);
    for (int s : nfa.accepting_states) {
        result.addEpsilonTransition(s + nfaOffset, accept);
    }
    return result;
}

NFA createLiteralNFA(const std::string& literal) {
    NFA nfa;
    if (literal.empty()) {
        int start = nfa.addState(true);
        nfa.start_state = start;
        return nfa;
    }
    
    // Create a chain of states that strictly matches the sequence
    int startState = nfa.addState(literal.length() == 0);  // Only accept empty string at start state
    nfa.start_state = startState;
    
    int currentState = startState;
    for (size_t i = 0; i < literal.length(); i++) {
        int nextState = nfa.addState(i == literal.length() - 1);  // Only last state is accepting
        nfa.addTransition(currentState, literal[i], nextState);
        currentState = nextState;
    }
    
    return nfa;
}

NFA regexToNFA(const std::string& regex) {
    std::string expanded = expandRepetition(regex);
    std::cout << "Expanded regex: " << expanded << std::endl;
    
    if (expanded.empty()) {
        NFA emptyNFA;
        int start = emptyNFA.addState();
        int accept = emptyNFA.addState(true);
        emptyNFA.start_state = start;
        emptyNFA.addEpsilonTransition(start, accept);
        return emptyNFA;
    }
    
    // Safety check for extremely complex regexes
    if (expanded.length() > 500) {
        std::cerr << "Warning: Regex pattern is very long, results may be unpredictable" << std::endl;
    }
    
    // Check if literal
    bool isLiteral = true;
    for (char c : expanded) {
        if (isOperator(c)) {
            isLiteral = false;
            break;
        }
    }
    if (isLiteral) {
        return createLiteralNFA(expanded);
    }
    
    try {
        // Add explicit concatenation and convert to postfix as before
        std::string processedRegex = addExplicitConcatenation(expanded);
        std::stack<char> operators;
        std::string postfix;
        for (char c : processedRegex) {
            if (isOperator(c)) {
                if (c == '(') {
                    operators.push(c);
                } else if (c == ')') {
                    while (!operators.empty() && operators.top() != '(') {
                        postfix += operators.top();
                        operators.pop();
                    }
                    if (!operators.empty()) operators.pop();
                } else {
                    int precedence = getPrecedence(c);
                    while (!operators.empty() && operators.top() != '(' && getPrecedence(operators.top()) >= precedence) {
                        postfix += operators.top();
                        operators.pop();
                    }
                    operators.push(c);
                }
            } else {
                postfix += c;
            }
        }
        while (!operators.empty()) {
            if (operators.top() == '(') {
                throw std::runtime_error("Mismatched parentheses in regex");
            }
            postfix += operators.top();
            operators.pop();
        }
        std::cout << "Postfix: " << postfix << std::endl;
        
        // ... existing postfix evaluation loop (handling . | * ? etc.)...
        std::stack<NFA> nfaStack;
        for (char c : postfix) {
            if (nfaStack.size() > 1000) {
                throw std::runtime_error("NFA construction error: Stack too deep, regex may be too complex");
            }
            if (c == '.') {
                if (nfaStack.size() < 2) {
                    throw std::runtime_error("Invalid regex syntax: not enough operands for concatenation");
                }
                NFA second = nfaStack.top(); nfaStack.pop();
                NFA first = nfaStack.top(); nfaStack.pop();
                nfaStack.push(concatenateNFAs(first, second));
            } else if (c == '|') {
                if (nfaStack.size() < 2) {
                    throw std::runtime_error("Invalid regex syntax: not enough operands for alternation");
                }
                NFA second = nfaStack.top(); nfaStack.pop();
                NFA first = nfaStack.top(); nfaStack.pop();
                nfaStack.push(alternateNFAs(first, second));
            } else if (c == '*') {
                if (nfaStack.empty()) {
                    throw std::runtime_error("Invalid regex syntax: not enough operands for Kleene star");
                }
                NFA operand = nfaStack.top(); nfaStack.pop();
                nfaStack.push(applyKleeneStar(operand));
            } else if (c == '?') {
                if (nfaStack.empty()) {
                    throw std::runtime_error("Invalid regex syntax: not enough operands for optional operator");
                }
                NFA operand = nfaStack.top(); nfaStack.pop();
                nfaStack.push(applyOptional(operand));
            } else {
                nfaStack.push(createBasicNFA(c));
            }
        }
        
        if (nfaStack.empty()) {
            throw std::runtime_error("Error: Failed to build NFA from regex pattern");
        }
        if (nfaStack.size() > 1) {
            throw std::runtime_error("Invalid regex syntax: too many operands left on stack");
        }
        NFA result = nfaStack.top();
        if (result.accepting_states.empty() && !result.states.empty()) {
            result.accepting_states.insert(result.states.back().id);
            result.states.back().is_accepting = true;
        }
        if (result.states.size() > 10000) {
            throw std::runtime_error("NFA has too many states (" +
                                  std::to_string(result.states.size()) +
                                  "), regex may be too complex");
        }
        
        // NEW: Normalize final state only if the regex contains closure operators.
        if (expanded.find('*') != std::string::npos || expanded.find('?') != std::string::npos) {
            // Clear all old accepting states.
            std::set<int> oldAccepting = result.accepting_states;
            for (int s : oldAccepting) {
                result.states[s].is_accepting = false;
            }
            result.accepting_states.clear();
            int finalState = result.addState(true);
            for (int acc : oldAccepting) {
                result.addEpsilonTransition(acc, finalState);
            }
            result.accepting_states.insert(finalState);
        }
        
        std::cout << "Created NFA with " << result.states.size() << " states" << std::endl;
        return result;
    }
    catch (const std::exception& e) {
        std::cerr << "Error in NFA construction: " << e.what() << std::endl;
        NFA fallbackNFA;
        int start = fallbackNFA.addState(true);
        fallbackNFA.start_state = start;
        fallbackNFA.addTransition(start, '0', start);
        fallbackNFA.addTransition(start, '1', start);
        return fallbackNFA;
    }
}

// ============ Implementazione della conversione da NFA a DFA ============
// Calcola la epsilon-closure di un insieme di stati NFA
std::set<int> epsilonClosure(const NFA& nfa, const std::set<int>& states) {
    std::set<int> result = states;
    std::stack<int> stack;
    
    for (int state : states) {
        stack.push(state);
    }
    
    while (!stack.empty()) {
        int state = stack.top();
        stack.pop();
        
        // Check if this state has any epsilon transitions
        auto epsilonIt = nfa.states[state].epsilon_transitions.find(0);
        if (epsilonIt != nfa.states[state].epsilon_transitions.end()) {
            for (int target : epsilonIt->second) {
                if (result.find(target) == result.end()) {
                    result.insert(target);
                    stack.push(target);
                }
            }
        }
    }
    
    return result;
}

// Calcola gli stati raggiungibili da un insieme di stati NFA con un dato simbolo
std::set<int> move(const NFA& nfa, const std::set<int>& states, char symbol) {
    std::set<int> result;
    
    for (int state : states) {
        auto it = nfa.states[state].transitions.find(symbol);
        if (it != nfa.states[state].transitions.end()) {
            result.insert(it->second.begin(), it->second.end());
        }
    }
    
    return result;
}

FSA NFAtoDFA(const NFA& nfa) {
    std::cout << "Converting NFA to DFA using subset construction" << std::endl;
    
    FSA dfa;
    std::map<std::set<int>, int> dfaStates;
    std::queue<std::set<int>> unmarkedStates;
    std::set<char> alphabet;
    
    // Safety check for extremely large NFAs
    if (nfa.states.size() > 10000) {
        throw std::runtime_error("NFA has too many states for DFA conversion");
    }
    
    // Collect alphabet symbols from NFA
    for (const auto& state : nfa.states) {
        for (const auto& [symbol, _] : state.transitions) {
            alphabet.insert(symbol);
        }
    }
    
    // Create a mapping from characters to symbol indices
    std::map<char, int> symbolToIndex;
    int index = 0;
    for (char symbol : alphabet) {
        symbolToIndex[symbol] = index++;
    }
    
    dfa.num_alphabet_symbols = alphabet.size();
    dfa.alphabet = std::vector<char>(alphabet.begin(), alphabet.end());
    
    // Start with epsilon-closure of the start state
    std::set<int> initialState = epsilonClosure(nfa, {nfa.start_state});
    dfaStates[initialState] = 0;  // Assign 0 as the ID of the initial DFA state
    unmarkedStates.push(initialState);
    dfa.start_state = 0;
    
    // Initialize transition function
    dfa.transition_function.clear();
    
    // Create a trap/dead state for invalid transitions
    int trapState = -1;
    
    // Process unmarked state sets
    while (!unmarkedStates.empty()) {
        std::set<int> currentStateSet = unmarkedStates.front();
        unmarkedStates.pop();
        
        int currentDfaState = dfaStates[currentStateSet];
        
        // Ensure there's room in the transition table
        while (dfa.transition_function.size() <= static_cast<size_t>(currentDfaState)) {
            dfa.transition_function.push_back(std::vector<int>(alphabet.size(), -1));
        }
        
        // For each symbol in the alphabet
        int symbolIdx = 0;
        for (char symbol : alphabet) {
            // Compute the next state set using more precise move and epsilon-closure
            std::set<int> nextStateSet = epsilonClosure(nfa, move(nfa, currentStateSet, symbol));
            
            if (nextStateSet.empty()) {
                // Create a trap state for invalid transitions
                if (trapState == -1) {
                    trapState = dfaStates.size();
                    dfaStates[std::set<int>()] = trapState;
                    
                    while (dfa.transition_function.size() <= static_cast<size_t>(trapState)) {
                        dfa.transition_function.push_back(std::vector<int>(alphabet.size(), trapState));
                    }
                    
                    // Make the trap state transition to itself for all symbols
                    for (size_t i = 0; i < alphabet.size(); i++) {
                        dfa.transition_function[trapState][i] = trapState;
                    }
                }
                dfa.transition_function[currentDfaState][symbolIdx] = trapState;
            } else {
                // Check if this is a new state set
                if (dfaStates.find(nextStateSet) == dfaStates.end()) {
                    int newDfaState = dfaStates.size();
                    dfaStates[nextStateSet] = newDfaState;
                    unmarkedStates.push(nextStateSet);
                    
                    // Ensure there's room in the transition table
                    while (dfa.transition_function.size() <= static_cast<size_t>(newDfaState)) {
                        dfa.transition_function.push_back(std::vector<int>(alphabet.size(), -1));
                    }
                }
                // Add the transition
                dfa.transition_function[currentDfaState][symbolIdx] = dfaStates[nextStateSet];
            }
            
            symbolIdx++;
        }
    }
    
    // Determine accepting states - only mark states containing NFA accepting states
    dfa.accepting_states.clear();
    for (const auto& [stateSet, dfaState] : dfaStates) {
        // Check if any NFA state in the set is accepting
        for (int nfaState : stateSet) {
            if (nfa.accepting_states.find(nfaState) != nfa.accepting_states.end()) {
                dfa.accepting_states.push_back(dfaState);
                break;
            }
        }
    }
    dfa.num_states = dfaStates.size();
    
    // Safety check for extremely large DFAs
    if (dfa.num_states > 10000) {
        throw std::runtime_error("DFA has too many states (" + 
                               std::to_string(dfa.num_states) + 
                               "), regex may be too complex");
    }
    
    std::cout << "Created DFA with " << dfa.num_states << " states" << std::endl;
    return dfa;
}

FSA FSAEngine::regexToDFA(const std::string& regex) {
    try {
        std::cout << "Starting regex to DFA conversion: " << regex << std::endl;
        NFA nfa = regexToNFA(regex);
        FSA dfa = NFAtoDFA(nfa);
        std::cout << "Conversion completed successfully" << std::endl;
        return dfa;
    } catch (const std::exception& e) {
        std::cerr << "Error in regexToDFA: " << e.what() << std::endl;
        // Return a simple default DFA if conversion fails
        FSA default_dfa;
        default_dfa.num_states = 2;
        default_dfa.num_alphabet_symbols = 2;
        default_dfa.start_state = 0;
        default_dfa.accepting_states = {1};
        default_dfa.transition_function.resize(2);
        for (int i = 0; i < 2; i++) {
            default_dfa.transition_function[i].resize(2, -1);
        }
        default_dfa.transition_function[0][0] = 0;
        default_dfa.transition_function[0][1] = 1;
        default_dfa.transition_function[1][0] = 0;
        default_dfa.transition_function[1][1] = 1;
        return default_dfa;
    }
}

bool FSAEngine::runDFA(const FSA& fsa, const std::string& input) {
    if (fsa.num_states == 0 || fsa.alphabet.empty()) {
        return false; // DFA vuoto rifiuta tutto
    }
    
    int currentState = fsa.start_state;
    
    // Caso speciale per stringa vuota: accetta se lo stato iniziale è accettante
    if (input.empty()) {
        return std::find(fsa.accepting_states.begin(), 
                         fsa.accepting_states.end(), 
                         currentState) != fsa.accepting_states.end();
    }
    
    // Elaborazione della stringa: per ogni simbolo si aggiorna lo stato corrente
    for (char c : input) {
        int symbolIdx = -1;
        for (size_t i = 0; i < fsa.alphabet.size(); i++) {
            if (c == fsa.alphabet[i]) {
                symbolIdx = i;
                break;
            }
        }
        
        // Se il simbolo non appartiene all'alfabeto, rifiuta l'input
        if (symbolIdx == -1) {
            return false;
        }
        
        // Verifica se esiste una transizione valida
        if (currentState >= static_cast<int>(fsa.transition_function.size()) ||
            symbolIdx >= static_cast<int>(fsa.transition_function[currentState].size()) ||
            fsa.transition_function[currentState][symbolIdx] < 0) {
            return false;
        }
        
        currentState = fsa.transition_function[currentState][symbolIdx];
    }
    
    // Accetta se lo stato finale è uno stato di accettazione
    return std::find(fsa.accepting_states.begin(), 
                     fsa.accepting_states.end(), 
                     currentState) != fsa.accepting_states.end();
}

// Note: CUDA-specific code (prepareGPUDFA, runBatchOnGPU, kernels) has been moved to fsa_cuda.cu