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
#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <string>
#include <numeric>  // Added for std::accumulate

// Add global debug log for conversion details.
static std::string conversionDebugLog;

std::string getConversionDebugLog() {
    return conversionDebugLog;
}

// Add this near the top of the file to collect debug information
std::string debugOutput;

void clearDebugOutput() {
    debugOutput.clear();
}

std::string getDebugOutput() {
    return debugOutput;
}

void addDebug(const std::string& message) {
    debugOutput += message + "\n";
}

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
    std::string result = "";
    for (size_t i = 0; i < regex.size(); ++i) {
        result += regex[i];
        if (i + 1 < regex.size()) {
            bool currIsOperand = (isalnum(regex[i]) || regex[i] == ')' || regex[i] == '*' || regex[i] == '?');
            bool nextIsOperand = (isalnum(regex[i + 1]) || regex[i + 1] == '(');

            // Insert concatenation ONLY if the current is an operand and the next is an operand,
            // OR if the current is a closing parenthesis and the next is an operand.
            if ((currIsOperand && nextIsOperand) || (regex[i] == ')' && nextIsOperand)) {
                result += '.';
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

// Improved preprocessRegex to handle negated classes, repetition, and optional operators
std::string preprocessRegex(const std::string &regex) {
    clearDebugOutput(); // Reset debug output
    std::string out;
    addDebug("Original regex: " + regex);
    
    for (size_t i = 0; i < regex.size(); ) {
        // Negated character classes: convert [^0] into ((1)) 
        if (i + 3 < regex.size() && regex[i]=='[' && regex[i+1]=='^' &&
            (regex[i+2]=='0' || regex[i+2]=='1') && regex[i+3]==']') {
            
            char operand = regex[i+2];
            char replacement = (operand=='0') ? '1' : '0';
            
            // If a Kleene star follows, output grouped version to enforce proper operator precedence.
            if (i + 4 < regex.size() && regex[i+4] == '*') {
                out += "((";
                out.push_back(replacement);
                out += ")*)";
                addDebug("Replaced [^" + std::string(1, operand) + "]* with ((replacement))*");
                i += 5; // Skip the entire [^x]*
            } else {
                // Otherwise, just substitute with its complement
                out.push_back(replacement);
                addDebug("Replaced [^" + std::string(1, operand) + "] with ((replacement))");
                i += 4;
            }
        }
        // Handle repetition {n}
        else if (i + 2 < regex.size() && regex[i] == '{' && isdigit(regex[i+1])) {
            size_t close = regex.find('}', i);
            if (close != std::string::npos) {
                std::string countStr = regex.substr(i+1, close-i-1);
                int count = std::stoi(countStr);
                addDebug("Found repetition {" + countStr + "}");
                std::string operand;
                // If the char before '{' is a closing parenthesis, extract the entire group.
                if (!out.empty() && out.back() == ')') {
                    int depth = 1;
                    size_t j = out.size() - 2; // start before last ')'
                    while (j < out.size() && depth > 0) { // j is unsigned so check for underflow using condition j < out.size()
                        if (out[j] == ')')
                            depth++;
                        else if (out[j] == '(')
                            depth--;
                        if(depth == 0) break;
                        if(j==0) break;
                        j--;
                    }
                    size_t groupStart = j;
                    operand = out.substr(groupStart);
                    // Remove the extracted group from out
                    out.erase(groupStart);
                } else if (!out.empty()) {
                    // Single character operand
                    operand = std::string(1, out.back());
                    out.pop_back();
                }
                // Append operand exactly count times
                for (int j = 0; j < count; j++) {
                    out += operand;
                }
                i = close + 1;
            } else {
                addDebug("Malformed repetition at position " + std::to_string(i));
                out.push_back(regex[i++]);
            }
        }
        // Handle optional operator '?'
        else if (i < regex.size() && regex[i] == '?') {
            size_t pos = out.rfind('(');
            if (pos != std::string::npos) {
                std::string group = out.substr(pos);
                out.erase(pos);
                out += "(" + group + "|)";
                addDebug("Replaced group " + group + "? with (" + group + "|)");
            } else if (!out.empty()) {
                char prev = out.back();
                out.pop_back();
                out += "(" + std::string(1, prev) + "|)";
                addDebug("Replaced '" + std::string(1, prev) + "?' with '(" + std::string(1, prev) + "|)'");
            }
            i++;
        }
        else {
            out.push_back(regex[i++]);
        }
    }
    
    addDebug("After preprocessing: " + out);
    return out;
}

// Fix the postfix evaluation to handle more complex expressions
NFA regexToNFA(const std::string& regex) {
    clearDebugOutput(); // Reset debug output
    
    std::string preprocessed = preprocessRegex(regex); // Preprocessing added
    conversionDebugLog += "DEBUG: Preprocessed regex: " + preprocessed + "\n";
    std::string expanded = expandRepetition(preprocessed);
    conversionDebugLog += "DEBUG: Expanded regex: " + expanded + "\n";

    addDebug("Expanded regex: " + expanded);

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
        conversionDebugLog += "DEBUG: Regex with explicit concatenation: " + processedRegex + "\n";
        addDebug("With explicit concatenation: " + processedRegex);
        std::stack<char> operators;
        std::string postfix;
        for (char c : processedRegex) {
            if (isOperator(c)) {
                if (c == '(') {
                    operators.push(c);
                    addDebug("Push ( to operator stack");
                } else if (c == ')') {
                    addDebug("Found ), popping operators until matching (");
                    while (!operators.empty() && operators.top() != '(') {
                        postfix += operators.top();
                        addDebug("Pop " + std::string(1, operators.top()) + " to postfix");
                        operators.pop();
                    }
                    if (!operators.empty()) {
                        addDebug("Pop ( from stack");
                        operators.pop();
                    } else {
                        addDebug("ERROR: Mismatched parentheses - missing (");
                    }
                } else {
                    int precedence = getPrecedence(c);
                    addDebug("Operator " + std::string(1, c) + " with precedence " + std::to_string(precedence));
                    while (!operators.empty() && operators.top() != '(' && getPrecedence(operators.top()) >= precedence) {
                        postfix += operators.top();
                        addDebug("Pop " + std::string(1, operators.top()) + " to postfix (higher precedence)");
                        operators.pop();
                    }
                    operators.push(c);
                    addDebug("Push " + std::string(1, c) + " to operator stack");
                }
            } else {
                postfix += c;
                addDebug("Add " + std::string(1, c) + " directly to postfix");
            }
        }
        while (!operators.empty()) {
            if (operators.top() == '(') {
                addDebug("ERROR: Mismatched parentheses - extra (");
                throw std::runtime_error("Mismatched parentheses in regex");
            }
            postfix += operators.top();
            addDebug("Pop remaining " + std::string(1, operators.top()) + " to postfix");
            operators.pop();
        }
        
        conversionDebugLog += "DEBUG: Postfix expression: " + postfix + "\n";
        addDebug("Final postfix expression: " + postfix);
        std::stack<NFA> nfaStack;
        for (char token : postfix) {
            if (isOperator(token)) {
                if (token == '*') {
                    if (nfaStack.empty())
                        throw std::runtime_error("Invalid regex syntax: not enough operands for Kleene star");
                    NFA operand = nfaStack.top();
                    nfaStack.pop();
                    nfaStack.push(applyKleeneStar(operand));
                } else if (token == '?') {
                    if (nfaStack.empty())
                        throw std::runtime_error("Invalid regex syntax: not enough operands for optional operator");
                    NFA operand = nfaStack.top();
                    nfaStack.pop();
                    nfaStack.push(applyOptional(operand));
                } else if (token == '.') {
                    // Concatenation: pop right first, then left,
                    // then concatenate left followed by right.
                    if (nfaStack.size() < 2)
                        throw std::runtime_error("Invalid regex syntax: not enough operands for concatenation");
                    NFA right = nfaStack.top(); nfaStack.pop();
                    NFA left = nfaStack.top(); nfaStack.pop();
                    // Use left-to-right concatenation order!
                    nfaStack.push(concatenateNFAs(left, right));
                } else if (token == '|') {
                    // Alternation: pop right then left, then alternate.
                    if (nfaStack.size() < 2)
                        throw std::runtime_error("Invalid regex syntax: not enough operands for alternation");
                    NFA right = nfaStack.top(); nfaStack.pop();
                    NFA left = nfaStack.top(); nfaStack.pop();
                    nfaStack.push(alternateNFAs(left, right));
                }
            } else {
                // Push NFA for literal token (or call createLiteralNFA)
                nfaStack.push(createLiteralNFA(std::string(1, token)));
            }
        }
        // Final processing: if more than one NFA remains, concatenate them left-to-right.
        if (nfaStack.empty()) {
            throw std::runtime_error("Error: Failed to build NFA from regex pattern");
        }
        if (nfaStack.size() > 1) {
            std::vector<NFA> nfaList;
            while (!nfaStack.empty()) {
                nfaList.push_back(nfaStack.top());
                nfaStack.pop();
            }
            std::reverse(nfaList.begin(), nfaList.end());
            NFA result = nfaList[0];
            for (size_t i = 1; i < nfaList.size(); ++i) {
                result = concatenateNFAs(result, nfaList[i]);
            }
            return result;
        }
        return nfaStack.top();
    }
    catch (const std::exception& e) {
        addDebug("ERROR: " + std::string(e.what()));
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
        
        // Get all epsilon transitions
        auto& transitions = nfa.states[state].epsilon_transitions;
        for (const auto& [eps, targets] : transitions) {
            for (int target : targets) {
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
    FSA dfa;
    std::map<std::set<int>, int> dfaStates;
    std::queue<std::set<int>> unmarkedStates;
    std::set<char> alphabet;
    
    // Safety check for extremely large NFA
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
    addDebug("Initial DFA state set: {" + 
           std::accumulate(initialState.begin(), initialState.end(), std::string(),
                         [](const std::string& a, int b) {
                             return a.empty() ? std::to_string(b) : a + "," + std::to_string(b);
                         }) + "}");
    dfaStates[initialState] = 0;  // Assign 0 as the ID of the initial DFA state
    unmarkedStates.push(initialState);
    dfa.start_state = 0;
    
    // Initialize transition function of the initial DFA state
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
            // Compute the next state set using move and epsilon-closure
            std::set<int> nextStateSet = epsilonClosure(nfa, move(nfa, currentStateSet, symbol));
            
            if (nextStateSet.empty()) {
                // Create a trap state for invalid transitions if needed
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
    
    // Determine accepting states - mark states containing NFA accepting states
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
    dfa.num_alphabet_symbols = alphabet.size();
    
    // Safety check for extremely large DFAs
    if (dfa.num_states > 10000) {
        throw std::runtime_error("DFA has too many states (" + 
                               std::to_string(dfa.num_states) + 
                               "), regex may be too complex");
    }
    
    return dfa;
}

FSA FSAEngine::regexToDFA(const std::string& regex) {
    conversionDebugLog.clear(); // Clear the log at the start of each conversion
    try {
        NFA nfa = regexToNFA(regex);
        FSA dfa = NFAtoDFA(nfa);
        conversionDebugLog += "DEBUG: DFA created with " + std::to_string(dfa.num_states) + " states\n";
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
