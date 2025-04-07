#include "regex_conversion.h" // Include the new header
#include "../include/fsa_definition.h" // Keep fsa_definition include
// Remove #include "fsa_engine.h" if FSAEngine class is no longer used for these functions
#include <stack>
#include <set>
#include <map>
#include <algorithm>
#include <iostream>
#include <sstream>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <string>
#include <numeric>  // Added for std::accumulate
#include <cctype>  // for isalnum, isdigit

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
    // Aggiunti '{' e '}' come operatori
    return (c == '|' || c == '.' || c == '*' || c == '(' || c == ')' || c == '?' || c == '+' || c == '{' || c == '}');
}

int getPrecedence(char op) {
    if (op == '|') return 1;
    if (op == '.') return 2;  // explicit concatenation
    if (op == '*' || op == '?' || op == '+') return 3; // Added '+' precedence
    return 0;
}

#include <cctype>  // for isalnum
std::string addExplicitConcatenation(const std::string& regex) {
    std::string result = "";
    for (size_t i = 0; i < regex.size(); ++i) {
        result += regex[i];
        if (i + 1 < regex.size()) {
            bool canConcatenateLeft = (isalnum(regex[i]) || regex[i] == ')' || regex[i] == '*' || regex[i] == '?' || regex[i] == '+' || regex[i] == '}');
            bool canConcatenateRight = (isalnum(regex[i + 1]) || regex[i + 1] == '(' || regex[i+1] == '[');

            if (canConcatenateLeft && canConcatenateRight) {
                result += '.';
            } else if (regex[i] == ']' && (isalnum(regex[i+1]) || regex[i+1] == '(')) {
                 result += '.';
            } else if (regex[i] == '*' && regex[i+1] == '(') {
                result += '.';
            } else if (regex[i] == ')' && regex[i+1] == '(') {
                result += '.';
            } else if (regex[i] == ')' && isalnum(regex[i+1])) {
                result += '.';
            } else if (isalnum(regex[i]) && regex[i+1] == '(') {
                result += '.';
            } else if (regex[i] == '}' && (isalnum(regex[i+1]) || regex[i+1] == '(')) {
                result += '.';
            }
        }
    }
    return result;
}

// NEW: Expand simple {n} operators in regex (supports single-character or parenthesized group operands)
std::string expandRepetition(const std::string regex) {
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
            if (count < 0) {
                throw std::runtime_error("Invalid repetition count: " + numStr);
            }

            // Determine the operand preceding '{'
            std::string operand;
            if (i > 0 && result.length() > 0) {
                if (result.back() == ')') {
                    // Find matching '(' in result
                    int depth = 0;
                    size_t pos = result.size()-1;
                    for (; pos < result.size(); pos--) {
                        if (result[pos] == ')') depth++;
                        else if (result[pos] == '(') {
                            depth--;
                            if (depth == 0) break;
                        }
                        if (pos == 0) break;
                    }
                    if (depth != 0)
                        throw std::runtime_error("Mismatched parentheses for repetition");
                    operand = result.substr(pos);
                    result.erase(pos);
                } else {
                    // Use the last character as operand
                    operand = result.substr(result.size()-1);
                    result.pop_back();
                }
            } else {
                throw std::runtime_error("No operand for repetition operator");
            }

            // Append operand count times
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

// Improved preprocessRegex to handle negated classes, repetition, and optional operators
std::string preprocessRegex(const std::string regex) {
    clearDebugOutput(); // Reset debug output
    std::string out;
    addDebug("Original regex: " + regex);

    for (size_t i = 0; i < regex.size(); ) {
        // Negated character classes: convert [^0] into (1) (not 0 means 1 in binary)
        if (i + 3 < regex.size() && regex[i]=='[' && regex[i+1]=='^' &&
            (regex[i+2]=='0' || regex[i+2]=='1') && regex[i+3]==']') {

            char operand = regex[i+2];
            char replacement = (operand=='0') ? '1' : '0';

            // Handle [^0]* pattern with correct grouping
            if (i + 4 < regex.size() && regex[i+4] == '*') {
                out += "(";
                out.push_back(replacement);
                out += ")*";
                addDebug("Replaced [^" + std::string(1, operand) + "]* with (" + std::string(1, replacement) + ")*");
                i += 5; // Skip the entire [^x]*
            } else {
                out += "(";
                out.push_back(replacement);
                out += ")";
                addDebug("Replaced [^" + std::string(1, operand) + "] with (" + std::string(1, replacement) + ")");
                i += 4;
            }
        }
        // Instead of transforming the optional operator, simply pass it through.
        else if (regex[i] == '?') {
            out.push_back(regex[i++]);
        }
        // Handle optional operator '?' by converting to alternation (a|ε)
        else if (regex[i] == '?') {
            // If the preceding char is ')', find the matching '(' using an int index.
            if (i > 0 && regex[i-1] == ')') {
                int depth = 0;
                int j = static_cast<int>(i) - 1; 
                for (; j >= 0; j--) {
                    if (regex[j] == ')') depth++;
                    else if (regex[j] == '(') {
                        depth--;
                        if (depth == 0) break;
                    }
                }
                if (depth != 0 || j < 0) {
                    throw std::runtime_error("Mismatched parentheses for optional operator");
                }
                std::string group = regex.substr(j, i - j);
                out = out.substr(0, out.size() - group.size()) + "(" + group + "|ε)";
                addDebug("Replaced group " + group + "? with (" + group + "|ε)");
            } else {
                // Regular case - replace 'a?' with '(a|ε)'
                char prev = out.back();
                out.pop_back();
                out += "(" + std::string(1, prev) + "|ε)";
                addDebug("Replaced '" + std::string(1, prev) + "?' with (" + std::string(1, prev) + "|ε)");
            }
            i++;
        }
         // Handle '+' operator (one or more occurrences) by converting to (aa*)
        else if (i > 0 && regex[i] == '+') {
            // If preceding char is ')', handle group
            if (i > 0 && regex[i-1] == ')') {
                int balance = 1;
                size_t j = i - 2;
                while (j < regex.size() && balance > 0) {
                    if (regex[j] == ')')
                        balance++;
                    else if (regex[j] == '(')
                        balance--;
                    if (balance == 0)
                        break;
                    if (j == 0)
                        break;
                    j--;
                }
                if (balance == 0) {
                    std::string group = regex.substr(j, i - j);
                    out = out.substr(0, out.size() - group.size()) + "(" + group + ")"+ "(" + group + ")*"; // Correct '+' to (group)(group)*
                    addDebug("Replaced group " + group + "+ with (" + group + ")(" + group + ")*");
                }
            } else {
                // Regular case - replace 'a+' with '(a)(a*)'
                char prev = out.back();
                out.pop_back();
                out += "(" + std::string(1, prev) + ")(" + std::string(1, prev) + ")*"; // Correct '+' to (a)(a*)
                addDebug("Replaced '" + std::string(1, prev) + "+' with (" + std::string(1, prev) + ")(" + std::string(1, prev) + ")*");
            }
            i++;
        }
        // Handle repetition {n} - leave for expandRepetition function
        else if (regex[i] == '{') {
            size_t close = regex.find('}', i);
            if (close != std::string::npos) {
                out += regex.substr(i, close - i + 1);
                addDebug("Preserved repetition token: " + regex.substr(i, close - i + 1));
                i = close + 1;
            } else {
                addDebug("Mismatched curly braces at position " + std::to_string(i));
                throw std::runtime_error("Mismatched curly braces in regex");
            }
        }
        else {
            out.push_back(regex[i++]);
        }
    }

    addDebug("After preprocessing: " + out);
    return out;
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
        result.addEpsilonTransition(acceptingState + firstOffset, second.start_state + secondOffset); // Corrected offset here
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
	// New Thompson construction for Kleene star:
	NFA result;
	// Create new start (non accepting) and new accept (accepting)
	int newStart = result.addState(false);
	int newAccept = result.addState(true);
	result.start_state = newStart;
	// Copy original NFA states; new ones will come after current states.
	int nfaOffset = result.states.size();
	result.copyStatesFrom(nfa, nfaOffset);
	// New start epsilon-transitions: to the original start and directly to new accept (empty string)
	result.addEpsilonTransition(newStart, nfa.start_state + nfaOffset);
	result.addEpsilonTransition(newStart, newAccept);
	// For each accepting state in original NFA,
	// add epsilon-transitions back to the original start and to new accept.
	for (int a : nfa.accepting_states) {
		result.addEpsilonTransition(a + nfaOffset, nfa.start_state + nfaOffset);
		result.addEpsilonTransition(a + nfaOffset, newAccept);
		// Clear the accepting flag on the copied state.
		result.states[a + nfaOffset].is_accepting = false;
	}
	return result;
}

NFA applyOptional(const NFA& nfa) {
	// New Thompson construction for optional operator:
	NFA result;
	// New start (non accepting) and new accept (accepting)
	int newStart = result.addState(false);
	int newAccept = result.addState(true);
	result.start_state = newStart;
	int nfaOffset = result.states.size();
	result.copyStatesFrom(nfa, nfaOffset);
	// Option: either skip operand or run operand.
	result.addEpsilonTransition(newStart, nfa.start_state + nfaOffset);
	result.addEpsilonTransition(newStart, newAccept);
	// For each accepting state in the operand, add an epsilon to new accept
	for (int a : nfa.accepting_states) {
		result.addEpsilonTransition(a + nfaOffset, newAccept);
		result.states[a + nfaOffset].is_accepting = false;
	}
	return result;
}

NFA applyPlusOperator(const NFA& nfa) {
    // Implemented using Kleene star and concatenation: N+ == NN*
    return concatenateNFAs(nfa, applyKleeneStar(nfa));
}

NFA createLiteralNFA(const std::string& literal) {
    NFA nfa;
    if(literal == "ε") {
        int start = nfa.addState(true);
        nfa.start_state = start;
        return nfa;
    }
    if (literal.empty()) {
        int start = nfa.addState(true);
        nfa.start_state = start;
        return nfa;
    }

    // Create a chain of states that strictly matches the sequence
    int startState = nfa.addState(false);  // Only accept empty string at start state
    nfa.start_state = startState;

    int currentState = startState;
    for (size_t i = 0; i < literal.length(); i++) {
        bool isLast = (i == literal.length() - 1);
        int nextState = nfa.addState(isLast);  // Only last state is accepting
        nfa.addTransition(currentState, literal[i], nextState);
        currentState = nextState;
    }

    return nfa;
}

// Fix the postfix evaluation to handle more complex expressions
// New function to check if a regex has known concatenation issues
bool hasRegexConcatenationIssue(const std::string& regex) {
    // These patterns have been identified as problematic
    static const std::vector<std::string> problematicPatterns = {
        "1*",
        "1[^0]*1", 
        "1{3}"
    };
    
    for (const auto& pattern : problematicPatterns) {
        if (regex == pattern) return true;
    }
    
    return false;
}

// Nuova funzione per applicare la ripetizione: costruisce NFA ripetuto n volte
NFA applyRepetition(const NFA& nfa, int n) {
    if (n <= 0) {
        NFA result;
        int start = result.addState(true);
        result.start_state = start;
        return result;
    }
    NFA result = nfa;
    for (int i = 1; i < n; ++i) {
        result = concatenateNFAs(result, nfa);
    }
    return result;
}

// Make debug messages in regexToNFA cleaner and more focused
NFA regexToNFA(const std::string& regex) {
    clearDebugOutput();

    std::string preprocessed = preprocessRegex(regex);
    conversionDebugLog += "DEBUG: Preprocessed regex: " + preprocessed + "\n";
    // NEW: Expand repetition tokens before further processing.
    std::string expanded = expandRepetition(preprocessed);
    conversionDebugLog += "DEBUG: Expanded regex: " + expanded + "\n";
    addDebug("Expanded regex: " + expanded);
    
    // If regex is empty, produce an NFA with a single accepting state.
    if (expanded.empty()) {
        NFA emptyNFA;
        int start = emptyNFA.addState(true);
        emptyNFA.start_state = start;
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
                } else { // Handle *, ?, +, |, . operators
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
        
        // Add debugging info for tracking the stack during evaluation
        std::stack<NFA> nfaStack;
        int tokenIndex = 0;
        
        for (size_t i = 0; i < postfix.size(); i++) {
            char token = postfix[i];
            tokenIndex++;
            // Log the stack size before each operation
            conversionDebugLog += "DEBUG: Processing token '" + std::string(1, token) + 
                                 "' (pos " + std::to_string(tokenIndex) + "/" + 
                                 std::to_string(postfix.length()) + ") - Stack size: " + 
                                 std::to_string(nfaStack.size()) + "\n";
            
            // Se il token inizia una ripetizione, leggiamo fino a '}'
            if (token == '{') {
                size_t close = postfix.find('}', i);
                if (close == std::string::npos) {
                    throw std::runtime_error("Mismatched braces in postfix expression");
                }
                int count = std::stoi(postfix.substr(i+1, close - i - 1));
                if (nfaStack.empty()) {
                    throw std::runtime_error("Invalid regex syntax: no operand for repetition operator at position " + std::to_string(i));
                }
                NFA operand = nfaStack.top();
                nfaStack.pop();
                nfaStack.push(applyRepetition(operand, count));
                addDebug("Applied repetition: {" + std::to_string(count) + "}");
                i = close; // salta il token di ripetizione
            }
            else if (isOperator(token)) {
                if (token == '*') {
                    if (nfaStack.empty()) {
                        std::string errMsg = "Invalid regex syntax: not enough operands for Kleene star in regex: " + regex;
                        conversionDebugLog += "ERROR: " + errMsg + "\n";
                        throw std::runtime_error(errMsg);
                    }
                    NFA operand = nfaStack.top();
                    nfaStack.pop();
                    nfaStack.push(applyKleeneStar(operand));
                } else if (token == '?') {
                    if (nfaStack.empty()) {
                        std::string errMsg = "Invalid regex syntax: not enough operands for optional operator in regex: " + regex;
                        conversionDebugLog += "ERROR: " + errMsg + "\n";
                        throw std::runtime_error(errMsg);
                    }
                    NFA operand = nfaStack.top();
                    nfaStack.pop();
                    nfaStack.push(applyOptional(operand));
                } else if (token == '+') {
                    if (nfaStack.empty()) {
                        std::string errMsg = "Invalid regex syntax: not enough operands for plus operator in regex: " + regex;
                        conversionDebugLog += "ERROR: " + errMsg + "\n";
                        throw std::runtime_error(errMsg);
                    }
                    NFA operand = nfaStack.top();
                    nfaStack.pop();
                    nfaStack.push(applyPlusOperator(operand));
                }
                else if (token == '.') {
                    // Concatenation: pop right first, then left,
                    // then concatenate left followed by right.
                    if (nfaStack.size() < 2) {
                        std::string errMsg = "Invalid regex syntax: not enough operands for concatenation in regex: " + regex + 
                                            "\nPostfix: " + postfix + 
                                            "\nCurrent token: " + std::string(1, token) + 
                                            " at position " + std::to_string(tokenIndex);
                        conversionDebugLog += "ERROR: " + errMsg + "\n";
                        throw std::runtime_error(errMsg);
                    }
                    NFA right = nfaStack.top(); nfaStack.pop();
                    NFA left = nfaStack.top(); nfaStack.pop();
                    // Use left-to-right concatenation order!
                    nfaStack.push(concatenateNFAs(left, right));
                } else if (token == '|') {
                    // Alternation: pop right then left, then alternate.
                    if (nfaStack.size() < 2) {
                        std::string errMsg = "Invalid regex syntax: not enough operands for alternation in regex: " + regex;
                        conversionDebugLog += "ERROR: " + errMsg + "\n";
                        throw std::runtime_error(errMsg);
                    }
                    NFA right = nfaStack.top(); nfaStack.pop();
                    NFA left = nfaStack.top(); nfaStack.pop();
                    nfaStack.push(alternateNFAs(left, right));
                }
             else {
                    throw std::runtime_error("Unsupported operator in postfix conversion");
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
        std::string errorMsg = e.what();
        addDebug("ERROR: " + errorMsg);
        conversionDebugLog += "ERROR: " + errorMsg + " in regex: " + regex + "\n";
        
        // Check for the specific concatenation error
        if (errorMsg.find("not enough operands for concatenation") != std::string::npos) {
            conversionDebugLog += "NOTE: This is a known issue with certain regex patterns.\n";
        }
        
        // ...existing code (create fallback NFA)...
        std::cerr << "Error in NFA construction: " << e.what() << std::endl;
        // Add the current regex to the log for debugging
        std::cerr << "Failed regex: \"" << regex << "\"" << std::endl;
        std::cerr << "Debug log: " << getDebugOutput() << std::endl;
        
        // Re-throw the exception to halt execution
        throw;
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
    
    // Fixed alphabet: always {'0', '1'}
    std::vector<char> alphabet = {'0', '1'};
    dfa.num_alphabet_symbols = 2;
    dfa.alphabet = alphabet;
    
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
                        dfa.transition_function.push_back(std::vector<int>(alphabet.size(), -1));
                    }
                    // Make the trap state transition to itself for all symbols
                    for (size_t i = 0; i < alphabet.size(); i++) {
                        dfa.transition_function[trapState][i] = trapState;
                    }
                }
                dfa.transition_function[currentDfaState][symbolIdx] = trapState;
            } else {
                if (dfaStates.find(nextStateSet) == dfaStates.end()) {
                    int newDfaState = dfaStates.size();
                    dfaStates[nextStateSet] = newDfaState;
                    unmarkedStates.push(nextStateSet);
                    while (dfa.transition_function.size() <= static_cast<size_t>(newDfaState)) {
                        dfa.transition_function.push_back(std::vector<int>(alphabet.size(), -1));
                    }
                }
                dfa.transition_function[currentDfaState][symbolIdx] = dfaStates[nextStateSet];
            }
            
            symbolIdx++;
        }
    }
    
    // Determine accepting states - mark states containing NFA accepting states
    dfa.accepting_states.clear();
    for (const auto& [stateSet, dfaState] : dfaStates) {
        for (int nfaState : stateSet) {
            if (nfa.accepting_states.find(nfaState) != nfa.accepting_states.end()) {
                dfa.accepting_states.push_back(dfaState);
                break;
            }
        }
    }
    // NEW: sort accepting states for consistent ordering
    std::sort(dfa.accepting_states.begin(), dfa.accepting_states.end());
    
    dfa.num_states = dfaStates.size();
    
    if (dfa.num_states > 10000) {
        throw std::runtime_error("DFA has too many states (" +
                               std::to_string(dfa.num_states) +
                               "), regex may be too complex");
    }
    
    return dfa;
}

// --- Main Conversion Function (within namespace) ---
namespace regex_conversion {

    // Move the implementation of regexToDFA here (remove from FSAEngine class if applicable)
    FSA regexToDFA(const std::string& regex) {
        conversionDebugLog.clear(); // Clear the log at the start of each conversion
        try {
            NFA nfa = ::regexToNFA(regex); // Call the global/static regexToNFA helper
            FSA dfa = ::NFAtoDFA(nfa);     // Call the global/static NFAtoDFA helper
            conversionDebugLog += "DEBUG: DFA created with " + std::to_string(dfa.num_states) + " states\n";
            return dfa;
        } catch (const std::exception& e) {
            // Throw an error if the regex is invalid
            std::string errorMsg = e.what();
            conversionDebugLog += "ERROR: " + errorMsg + "\n";
            std::cerr << "Error in regexToDFA: " << errorMsg << std::endl;
            throw std::runtime_error("Invalid regex: " + errorMsg);
        }
    }

    // Implementations for the debug functions if they weren't static before
    std::string getConversionDebugLog() {
        return ::conversionDebugLog; // Access the static global variable
    }

    void clearDebugOutput() {
        ::debugOutput.clear();
    }

    std::string getDebugOutput() {
        return ::debugOutput;
    }

    void addDebug(const std::string& message) {
        ::debugOutput += message + "\n";
    }

} // namespace regex_conversion

// --- DFA Execution Function ---
// Remove the unused static runDFA function
/* static bool runDFA(const FSA& fsa, const std::string& input) {
    // ... implementation removed ...
} */


// --- C Interface Functions ---
// Keep implementations for regex_to_fsa, free_fsa, fsa_to_data, free_fsa_data
// Ensure they call the namespaced regex_conversion::regexToDFA
extern "C" FSA* regex_to_fsa(const char* regex) {
    try {
        // Use the namespaced function
        FSA fsa = regex_conversion::regexToDFA(regex);
        // Alloca memoria per FSA e copia i dati
        FSA* fsa_ptr = new FSA;
        *fsa_ptr = fsa;
        return fsa_ptr;
    } catch (const std::exception& e) {
        std::cerr << "Errore in regex_to_fsa: " << e.what() << std::endl;
        return nullptr;
    }
}

extern "C" void free_fsa(FSA* fsa) {
    delete fsa;
}

extern "C" FSAData* fsa_to_data(const FSA& fsa) {
    FSAData* data = new FSAData;
    data->num_states = fsa.num_states;
    data->num_alphabet_symbols = fsa.num_alphabet_symbols;
    data->start_state = fsa.start_state;

    // Appiattisci transition_function
    data->transition_function_size = fsa.num_states * fsa.num_alphabet_symbols;
    data->transition_function = new int[data->transition_function_size];
    // Initialize with -1 or a specific trap state value if applicable
    std::fill_n(data->transition_function, data->transition_function_size, -1);
    for (int state = 0; state < fsa.num_states; ++state) {
        // Check state bounds for transition_function vector
        if (state < fsa.transition_function.size()) {
            for (int symbol = 0; symbol < fsa.num_alphabet_symbols; ++symbol) {
                 // Check symbol bounds for the inner vector
                 if (symbol < fsa.transition_function[state].size()) {
                     data->transition_function[state * fsa.num_alphabet_symbols + symbol] = fsa.transition_function[state][symbol];
                 }
            }
        }
    }


    // Copia accepting_states
    data->accepting_states_size = fsa.accepting_states.size();
    data->accepting_states = new int[data->accepting_states_size];
    std::copy(fsa.accepting_states.begin(), fsa.accepting_states.end(), data->accepting_states);

    // Copia alphabet
    data->alphabet_size = fsa.alphabet.size();
    data->alphabet = new char[data->alphabet_size];
    std::copy(fsa.alphabet.begin(), fsa.alphabet.end(), data->alphabet);

    return data;
}

extern "C" void free_fsa_data(FSAData* data) {
    if (data) {
        delete[] data->transition_function;
        delete[] data->accepting_states;
        delete[] data->alphabet;
        delete data;
    }
}