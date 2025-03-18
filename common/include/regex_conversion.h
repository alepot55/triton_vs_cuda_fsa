#ifndef REGEX_CONVERSION_H
#define REGEX_CONVERSION_H

#include <string>

// Declarations for the regex conversion debug functions
std::string getConversionDebugLog();
std::string getDebugOutput();
void clearDebugOutput();
void addDebug(const std::string& message);

// New function to check if a regex has concatenation issues
bool hasRegexConcatenationIssue(const std::string& regex);

#endif // REGEX_CONVERSION_H
