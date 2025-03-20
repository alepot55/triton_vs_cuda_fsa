"""
Common test file parser.
Restituisce una lista di dizionari con i campi "name", "regex", "input" ed eventualmente "expected".
"""
def parse_test_file(filename):
    tests = []
    try:
        with open(filename, 'r') as file:
            current = {}
            for line in file:
                line = line.strip()
                if not line or line.startswith('#'):
                    continue
                if line.startswith('[') and line.endswith(']'):
                    if current.get("regex"):
                        tests.append(current)
                    current = {"name": line[1:-1]}
                elif '=' in line:
                    key, value = line.split('=', 1)
                    current[key.strip()] = value.strip()
            if current.get("regex"):
                tests.append(current)
    except Exception as e:
        print(f"Error parsing test file: {e}")
    return tests

if __name__ == "__main__":
    import sys
    filename = sys.argv[1] if len(sys.argv) > 1 else "test_cases.txt"
    parsed = parse_test_file(filename)
    print(parsed)
