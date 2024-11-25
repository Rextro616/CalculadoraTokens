from flask import Flask, render_template, request, jsonify
from lark import Lark, Transformer, Tree, Token
import ply.lex as lex
import matplotlib
matplotlib.use('Agg')  # Para evitar conflictos con GUI
import matplotlib.pyplot as plt
import os

app = Flask(__name__)

# Gramática para la calculadora
grammar = """
?start: expr
?expr: term
     | expr "+" term   -> add
     | expr "-" term   -> sub
?term: factor
     | term "*" factor -> mul
     | term "/" factor -> div
?factor: NUMBER        -> number
       | "(" expr ")"  -> parens
NUMBER: /\d+(\.\d+)?/
%ignore " "            
"""

# Analizador (Parser)
parser = Lark(grammar, start='start', parser='lalr')

# Transformador para evaluar expresiones y generar resultados
class CalculateTree(Transformer):
    def number(self, args):
        return float(args[0])

    def parens(self, args):
        return args[0]
    
    def add(self, args):
        return args[0] + args[1]

    def sub(self, args):
        return args[0] - args[1]

    def mul(self, args):
        return args[0] * args[1]

    def div(self, args):
        if args[1] == 0:
            raise ValueError("División por cero")
        return args[0] / args[1]

# Lexer con PLY
tokens = ('NUMERO', 'SUMA', 'RESTA', 'MULTIPLICACION', 'DIVISION')

t_SUMA = r'\+'
t_RESTA = r'-'
t_MULTIPLICACION = r'\*'
t_DIVISION = r'/'

def t_NUMERO(t):
    r'\d+(\.\d+)?'
    t.value = float(t.value) if '.' in t.value else int(t.value)
    return t

t_ignore = ' \t'

def t_error(t):
    raise SyntaxError(f"Illegal character '{t.value[0]}' at position {t.lexpos}")

lexer = lex.lex()

# Análisis de tokens
def analyze_tokens(expression):
    lexer.input(expression)
    total_numbers = 0
    total_operators = 0
    tokens_list = []

    for token in lexer:
        tokens_list.append({'type': token.type, 'value': token.value})
        if token.type == 'NUMERO':
            total_numbers += 1
        elif token.type in ('SUMA', 'RESTA', 'MULTIPLICACION', 'DIVISION'):
            total_operators += 1

    return {
        'total_numbers': total_numbers,
        'total_operators': total_operators,
        'tokens_list': tokens_list
    }

# Dibujar árbol sintáctico
def draw_tree(tree, x=0, y=0, x_offset=1.5, y_offset=1, graph=None):
    """Dibuja un árbol utilizando Matplotlib."""
    if graph is None:
        fig, graph = plt.subplots(figsize=(10, 6))

    if isinstance(tree, tuple):  # Nodo con hijos
        label = str(tree[0])
        graph.text(x, y, label, ha='center', bbox=dict(boxstyle="round", facecolor="lightblue"))
        for i, child in enumerate(tree[1], start=1):
            new_x = x + x_offset * (i - (len(tree[1]) / 2))  # Espaciado entre hijos
            new_y = y - y_offset
            graph.plot([x, new_x], [y - 0.1, new_y + 0.1], color="black")  # Línea entre nodos
            draw_tree(child, new_x, new_y, x_offset / 2, y_offset, graph)
    else:  # Nodo hoja
        graph.text(x, y, str(tree), ha='center', bbox=dict(boxstyle="round", facecolor="lightgreen"))

    graph.axis("off")
    return graph

# Función para limpiar el árbol
def clean_tree(tree):
    if isinstance(tree, Tree):
        return (tree.data, [clean_tree(child) for child in tree.children])
    elif isinstance(tree, Token):
        return float(tree) if tree.type == "NUMBER" else str(tree)
    else:
        return tree

@app.route("/", methods=["GET", "POST"])
def index():
    tree_path = None
    error = None
    result = None
    tokens_data = None

    if request.method == "POST":
        expression = request.form['expression']
        try:
            # Generar el árbol sintáctico
            tree = parser.parse(expression)
            transformer = CalculateTree()
            result = transformer.transform(tree)
            parsed_tree = clean_tree(tree)

            # Dibujar el árbol
            fig = draw_tree(parsed_tree)
            tree_path = "static/tree.png"
            plt.savefig(tree_path)
            plt.close()

            # Analizar tokens
            tokens_data = analyze_tokens(expression)

        except Exception as e:
            error = f"Error al procesar la expresión: {e}"

    return render_template("index.html", tree_path=tree_path, result=result, error=error, tokens_data=tokens_data)

if __name__ == "__main__":
    app.run(debug=True)
