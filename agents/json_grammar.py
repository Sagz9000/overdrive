"""
Constrained JSON Grammar for Tool Calls
==========================================
Implements the PRD's "constrained decoding" requirement (Section 4.2).

Provides a GBNF grammar string that forces all LLM output into structured
JSON tool-call format. This is used with llama-cpp-python's native grammar
parameter to guarantee valid output at the token level — not via prompting.

The grammar enforces one of two output formats:
1. Tool call:     {"tool": "<name>", "arguments": {...}}
2. With reasoning: {"reasoning": "...", "tool": "<name>", "arguments": {...}}
"""

# GBNF grammar for llama.cpp constrained decoding
# This grammar forces the model to output valid JSON matching the tool-call schema.
TOOL_CALL_GRAMMAR = r'''
root ::= ws tool-call ws

tool-call ::= "{" ws reasoning-field? tool-field "," ws arguments-field ws "}"

reasoning-field ::= "\"reasoning\"" ws ":" ws string "," ws

tool-field ::= "\"tool\"" ws ":" ws string

arguments-field ::= "\"arguments\"" ws ":" ws object

object ::= "{" ws (pair ("," ws pair)*)? ws "}"

pair ::= string ws ":" ws value

array ::= "[" ws (value ("," ws value)*)? ws "]"

value ::= string | number | object | array | "true" | "false" | "null"

string ::= "\"" characters "\""

characters ::= character*

character ::= [^"\\] | "\\" escape-char

escape-char ::= ["\\\/bfnrt] | "u" hex hex hex hex

hex ::= [0-9a-fA-F]

number ::= "-"? integer fraction? exponent?

integer ::= "0" | [1-9] [0-9]*

fraction ::= "." [0-9]+

exponent ::= [eE] [+-]? [0-9]+

ws ::= [ \t\n\r]*
'''

# Simpler grammar that allows plain text reasoning followed by a JSON block
# Use this if the model struggles with the strict JSON-only format.
HYBRID_GRAMMAR = r'''
root ::= ws (tool-call | reasoning-then-call) ws

tool-call ::= "{" ws tool-field "," ws arguments-field ws "}"

reasoning-then-call ::= "{" ws "\"reasoning\"" ws ":" ws string "," ws tool-field "," ws arguments-field ws "}"

tool-field ::= "\"tool\"" ws ":" ws string

arguments-field ::= "\"arguments\"" ws ":" ws object

object ::= "{" ws (pair ("," ws pair)*)? ws "}"

pair ::= string ws ":" ws value

array ::= "[" ws (value ("," ws value)*)? ws "]"

value ::= string | number | object | array | "true" | "false" | "null"

string ::= "\"" characters "\""

characters ::= character*

character ::= [^"\\] | "\\" escape-char

escape-char ::= ["\\\/bfnrt] | "u" hex hex hex hex

hex ::= [0-9a-fA-F]

number ::= "-"? integer fraction? exponent?

integer ::= "0" | [1-9] [0-9]*

fraction ::= "." [0-9]+

exponent ::= [eE] [+-]? [0-9]+

ws ::= [ \t\n\r]*
'''


def get_grammar_string(strict: bool = True) -> str:
    """
    Get the GBNF grammar string for constrained decoding.

    Args:
        strict: If True, use the strict tool-call-only grammar.
                If False, use the hybrid grammar that allows reasoning.

    Returns:
        GBNF grammar string compatible with llama-cpp-python.
    """
    return TOOL_CALL_GRAMMAR if strict else HYBRID_GRAMMAR


def create_llama_grammar(strict: bool = True):
    """
    Create a LlamaGrammar object for use with llama-cpp-python.

    Returns None if llama_cpp is not available.
    """
    try:
        from llama_cpp import LlamaGrammar
        grammar_str = get_grammar_string(strict)
        return LlamaGrammar.from_string(grammar_str)
    except ImportError:
        print("Warning: llama_cpp not available, grammar constraints disabled.")
        return None
    except Exception as e:
        print(f"Warning: Failed to create grammar: {e}")
        return None


def save_grammar_file(output_path: str = None, strict: bool = True):
    """Save the grammar to a .gbnf file for use with llama.cpp CLI."""
    import os
    if output_path is None:
        output_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            'tool_call.gbnf'
        )

    grammar_str = get_grammar_string(strict)
    with open(output_path, 'w') as f:
        f.write(grammar_str)

    print(f"Grammar saved to: {output_path}")


if __name__ == "__main__":
    # Generate the .gbnf file
    save_grammar_file()
    print("Grammar file generated successfully.")
