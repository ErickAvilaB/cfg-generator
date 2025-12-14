#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""cfg_generator

Genera una imajen PNG con el Control Flow Graph (CFG) de un arhcivo Python.

Uso:
    python3 cfg_generator.py ruta/al/programa.py

Salida:
    Crea un archivo con el mismo nombre pero extension .png en el mismo directorio.

Sugerencia:
    Para autoanalisis ejecuta:
        python3 cfg_generator.py cfg_generator.py
"""


from __future__ import annotations

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple

import pygraphviz


def unparse(nodo: ast.AST) -> str:
    """Convierte un nodo AST a texto."""
    return ast.unparse(nodo).strip()


class NodoCFG(dict):
    """Nodo del CFG.

    Mantiene referencias a padres e hijos, y guarda el AST asociado.
    """

    contador: int = 0
    cache: Dict[int, "NodoCFG"] = {}

    def __init__(self, padres=None, nodo_ast: Optional[ast.AST] = None):
        super().__init__()

        # En el CFG se usa una tupla (lista_de_padres, True/False) para marcar
        # el tipo de rama en if/while/for. Conservamos esa convencion.
        self.tipo_rama = ""
        if isinstance(padres, tuple) and len(padres) == 2:
            padres, self.tipo_rama = padres

        self.padres: List[NodoCFG] = list(padres or [])
        self.hijos: List[NodoCFG] = []
        self.llamadas: List[str] = []
        self.nodo_ast: ast.AST = nodo_ast if nodo_ast is not None else ast.parse(
            "pass").body[0]

        self.id: int = NodoCFG.contador
        NodoCFG.cache[self.id] = self
        NodoCFG.contador += 1

    def linea(self) -> int:
        return int(getattr(self.nodo_ast, "lineno", 0) or 0)

    def fuente(self) -> str:
        return unparse(self.nodo_ast).strip()

    def agregar_hijo(self, hijo: "NodoCFG") -> None:
        # NodoCFG hereda de dict. Dos nodos "vacios" comparan iguales ({ } == { }),
        # asi que no se debe usar `in` directo para evitar duplicados.
        if not any(h is hijo for h in self.hijos):
            self.hijos.append(hijo)

    def agregar_padre(self, padre: "NodoCFG") -> None:
        if not any(p is padre for p in self.padres):
            self.padres.append(padre)

    def agregar_padres(self, padres) -> None:
        """Agrega varios padres al nodo.

        Acepta una lista de nodos, o la tupla (lista, True/False) usada para ramas.
        """

        if isinstance(padres, tuple) and len(padres) == 2:
            padres = padres[0]

        for p in padres or []:
            self.agregar_padre(p)

    def agregar_llamada(self, nombre: str) -> None:
        self.llamadas.append(nombre)


class GeneradorCFG:
    """Construye el CFG recorriendo el AST."""

    def __init__(self) -> None:
        self.nodo_inicio = NodoCFG(
            padres=[], nodo_ast=ast.parse("start").body[0])
        self.nodo_inicio.nodo_ast.lineno = 0
        self.funciones: Dict[str, List[NodoCFG]] = {}
        self.funcion_por_linea: Dict[int, str] = {}

    def parsear(self, codigo: str) -> ast.AST:
        return ast.parse(codigo)

    def visitar(self, nodo: Optional[ast.AST], padres):
        if nodo is None:
            return padres
        nombre = f"en_{nodo.__class__.__name__.lower()}"
        manejador = getattr(self, nombre, None)
        if manejador:
            return manejador(nodo, padres)

        # Fallback: evita que una tupla (padres, rama) se propague como si fuera lista de nodos.
        if isinstance(nodo, ast.stmt):
            return [NodoCFG(padres=padres, nodo_ast=nodo)]
        return padres[0] if isinstance(padres, tuple) and len(padres) == 2 else padres

    def en_module(self, nodo: ast.Module, padres):
        p = padres
        for sentencia in nodo.body:
            p = self.visitar(sentencia, p)
        return p

    def en_pass(self, nodo: ast.Pass, padres):
        return [NodoCFG(padres=padres, nodo_ast=nodo)]

    def en_assign(self, nodo: ast.Assign, padres):
        if len(nodo.targets) > 1:
            raise NotImplementedError("Asignaciones paralelas no soportadas")
        p = [NodoCFG(padres=padres, nodo_ast=nodo)]
        return self.visitar(nodo.value, p)

    def en_expr(self, nodo: ast.Expr, padres):
        p = [NodoCFG(padres=padres, nodo_ast=nodo)]
        return self.visitar(nodo.value, p)

    def en_binop(self, nodo: ast.BinOp, padres):
        p = self.visitar(nodo.left, padres)
        return self.visitar(nodo.right, p)

    def en_unaryop(self, nodo: ast.UnaryOp, padres):
        return self.visitar(nodo.operand, padres)

    def en_compare(self, nodo: ast.Compare, padres):
        p = self.visitar(nodo.left, padres)
        return self.visitar(nodo.comparators[0], p)

    def en_if(self, nodo: ast.If, padres):
        nodo_test = NodoCFG(
            padres=padres,
            nodo_ast=ast.parse(f"_if: {unparse(nodo.test)}").body[0],
        )
        ast.copy_location(nodo_test.nodo_ast, nodo.test)

        test = self.visitar(nodo.test, [nodo_test])

        # Si la rama esta vacia, el flujo sale directo desde el test.
        fin_true = test
        if nodo.body:
            p_true = (test, True)
            for s in nodo.body:
                p_true = self.visitar(s, p_true)
            fin_true = p_true

        fin_false = test
        if nodo.orelse:
            p_false = (test, False)
            for s in nodo.orelse:
                p_false = self.visitar(s, p_false)
            fin_false = p_false

        return list(fin_true) + list(fin_false)

    def en_while(self, nodo: ast.While, padres):
        nodo_test = NodoCFG(
            padres=padres,
            nodo_ast=ast.parse(f"_while: {unparse(nodo.test)}").body[0],
        )
        ast.copy_location(nodo_test.nodo_ast, nodo.test)

        nodo_test.nodos_salida = []  # para break
        test = self.visitar(nodo.test, [nodo_test])

        if nodo.body:
            p = (test, True)
            for s in nodo.body:
                p = self.visitar(s, p)
            nodo_test.agregar_padres(p)

        # Al salir por la condicion falsa, los padres son los nodos del test.
        return list(nodo_test.nodos_salida) + list(test)

    def en_for(self, nodo: ast.For, padres):
        nodo_test = NodoCFG(
            padres=padres,
            nodo_ast=ast.parse(
                f"_for: True if {unparse(nodo.iter)} else False").body[0],
        )
        ast.copy_location(nodo_test.nodo_ast, nodo)

        nodo_test.nodos_salida = []
        test = self.visitar(nodo.iter, [nodo_test])

        extraer = NodoCFG(
            padres=[nodo_test],
            nodo_ast=ast.parse(
                f"{unparse(nodo.target)} = {unparse(nodo.iter)}.shift()"
            ).body[0],
        )
        ast.copy_location(extraer.nodo_ast, nodo_test.nodo_ast)

        if nodo.body:
            p = [extraer]
            for s in nodo.body:
                p = self.visitar(s, p)
            nodo_test.agregar_padres(p)
        else:
            nodo_test.agregar_padres([extraer])

        return list(nodo_test.nodos_salida) + list(test)

    def en_break(self, nodo: ast.Break, padres):
        padres_lista = padres[0] if isinstance(
            padres, tuple) and len(padres) == 2 else padres
        padre = padres_lista[0] if padres_lista else None
        if padre is None:
            return []
        while not hasattr(padre, "nodos_salida"):
            padre = padre.padres[0]

        actual = NodoCFG(padres=padres, nodo_ast=nodo)
        padre.nodos_salida.append(actual)
        return []

    def en_continue(self, nodo: ast.Continue, padres):
        padres_lista = padres[0] if isinstance(
            padres, tuple) and len(padres) == 2 else padres
        padre = padres_lista[0] if padres_lista else None
        if padre is None:
            return []
        while not hasattr(padre, "nodos_salida"):
            padre = padre.padres[0]

        actual = NodoCFG(padres=padres, nodo_ast=nodo)
        padre.agregar_padre(actual)
        return []

    def en_return(self, nodo: ast.Return, padres):
        padre = padres[0][0] if isinstance(padres, tuple) else padres[0]

        valor = self.visitar(nodo.value, padres)
        while not hasattr(padre, "nodos_return"):
            padre = padre.padres[0]

        actual = NodoCFG(padres=valor, nodo_ast=nodo)
        padre.nodos_return.append(actual)
        return []

    def en_functiondef(self, nodo: ast.FunctionDef, padres):
        firma = ", ".join([a.arg for a in nodo.args.args])

        nodo_entrada = NodoCFG(padres=[], nodo_ast=ast.parse(
            f"enter: {nodo.name}({firma})").body[0])
        nodo_entrada.enlace_callee = True
        ast.copy_location(nodo_entrada.nodo_ast, nodo)
        nodo_entrada.nodos_return = []

        nodo_salida = NodoCFG(padres=[], nodo_ast=ast.parse(
            f"exit: {nodo.name}({firma})").body[0])
        nodo_salida.nodo_salida_fn = True
        ast.copy_location(nodo_salida.nodo_ast, nodo)

        p = [nodo_entrada]
        for s in nodo.body:
            p = self.visitar(s, p)

        for n in p:
            if n not in nodo_entrada.nodos_return:
                nodo_entrada.nodos_return.append(n)

        for n in nodo_entrada.nodos_return:
            nodo_salida.agregar_padre(n)

        self.funciones[nodo.name] = [nodo_entrada, nodo_salida]
        self.funcion_por_linea[nodo_entrada.linea()] = nodo.name
        return padres

    def en_call(self, nodo: ast.Call, padres):
        def nombre_funcion(n: ast.AST) -> str:
            if isinstance(n, ast.Name):
                return n.id
            if isinstance(n, ast.Attribute):
                return n.attr
            if isinstance(n, ast.Call):
                return nombre_funcion(n.func)
            raise TypeError(str(type(n)))

        padres_lista = padres[0] if isinstance(
            padres, tuple) and len(padres) == 2 else padres

        p = padres_lista
        for arg in nodo.args:
            p = self.visitar(arg, p)

        nombre = nombre_funcion(nodo.func)
        if padres_lista:
            padres_lista[0].agregar_llamada(nombre)

        for n in p or []:
            n.enlace_llamada = 0
        return p

    def _funcion_de(self, nodo: NodoCFG) -> str:
        if nodo.linea() in self.funcion_por_linea:
            return self.funcion_por_linea[nodo.linea()]
        if not nodo.padres:
            self.funcion_por_linea[nodo.linea()] = ""
            return ""
        valor = self._funcion_de(nodo.padres[0])
        self.funcion_por_linea[nodo.linea()] = valor
        return valor

    def _actualizar_hijos(self) -> None:
        for _, nodo in NodoCFG.cache.items():
            for p in nodo.padres:
                p.agregar_hijo(nodo)

    def _actualizar_funciones(self) -> None:
        for _, nodo in NodoCFG.cache.items():
            self._funcion_de(nodo)

    def _enlazar_funciones(self) -> None:
        for _, nodo in NodoCFG.cache.items():
            if not nodo.llamadas:
                continue

            for llamada in nodo.llamadas:
                if llamada not in self.funciones:
                    continue

                entrada, salida = self.funciones[llamada]
                entrada.agregar_padre(nodo)

                if nodo.hijos:
                    nodo.enlace_llamada += 1
                    for h in nodo.hijos:
                        h.agregar_padre(salida)

    def generar(self, codigo: str) -> None:
        arbol = self.parsear(codigo)
        finales = self.visitar(arbol, [self.nodo_inicio])

        self.nodo_fin = NodoCFG(
            padres=finales, nodo_ast=ast.parse("stop").body[0])
        ast.copy_location(self.nodo_fin.nodo_ast, self.nodo_inicio.nodo_ast)

        self._actualizar_hijos()
        self._actualizar_funciones()
        self._enlazar_funciones()


def _normalizar_etiqueta(texto: str) -> str:
    for clave in ["if", "while", "for", "elif"]:
        texto = re.sub(rf"^_{clave}:", f"{clave}:", texto)
    return texto


def _resumir_texto(texto: str, limite: int = 90) -> str:
    """Reduce etiquetas largas para que el grafo sea legible y rapido de dibujar."""

    plano = re.sub(r"\s+", " ", texto.replace("\n", " ")).strip()
    if len(plano) <= limite:
        return plano
    return plano[: max(0, limite - 3)] + "..."


def _nombres_leidos(nodo: ast.AST) -> Set[str]:
    nombres: Set[str] = set()
    for sub in ast.walk(nodo):
        if isinstance(sub, ast.Name) and isinstance(sub.ctx, ast.Load):
            nombres.add(sub.id)
    return nombres


def _nombres_escritos(nodo: ast.AST) -> Set[str]:
    nombres: Set[str] = set()
    if isinstance(nodo, ast.Assign) and len(nodo.targets) == 1 and isinstance(nodo.targets[0], ast.Name):
        nombres.add(nodo.targets[0].id)
    if isinstance(nodo, ast.AnnAssign) and isinstance(nodo.target, ast.Name):
        nombres.add(nodo.target.id)
    return nombres


def _evaluar_constante(expr: ast.AST) -> Optional[object]:
    """Evalua una expresion simple cuando es claramente constante.

    No intenta resolver nombres; si hay variables, regresa None.
    """
    def ev(e: ast.AST) -> Optional[object]:
        if isinstance(e, ast.Constant):
            return e.value
        if isinstance(e, ast.UnaryOp):
            v = ev(e.operand)
            if v is None:
                return None
            if isinstance(e.op, ast.Not):
                return not bool(v)
            if isinstance(e.op, ast.USub):
                return -v  # type: ignore[operator]
            return None
        if isinstance(e, ast.BinOp):
            a, b = ev(e.left), ev(e.right)
            if a is None or b is None:
                return None
            if isinstance(e.op, ast.Add):
                return a + b  # type: ignore[operator]
            if isinstance(e.op, ast.Sub):
                return a - b  # type: ignore[operator]
            if isinstance(e.op, ast.Mult):
                return a * b  # type: ignore[operator]
            if isinstance(e.op, ast.FloorDiv):
                return a // b  # type: ignore[operator]
            return None
        if isinstance(e, ast.BoolOp):
            vals = [ev(v) for v in e.values]
            if any(v is None for v in vals):
                return None
            if isinstance(e.op, ast.And):
                out = True
                for v in vals:
                    out = out and bool(v)
                return out
            if isinstance(e.op, ast.Or):
                out = False
                for v in vals:
                    out = out or bool(v)
                return out
        if isinstance(e, ast.Compare) and len(e.ops) == 1 and len(e.comparators) == 1:
            a = ev(e.left)
            b = ev(e.comparators[0])
            if a is None or b is None:
                return None
            op = e.ops[0]
            if isinstance(op, ast.Eq):
                return a == b
            if isinstance(op, ast.NotEq):
                return a != b
            if isinstance(op, ast.Gt):
                return a > b  # type: ignore[operator]
            if isinstance(op, ast.Lt):
                return a < b  # type: ignore[operator]
            if isinstance(op, ast.GtE):
                return a >= b  # type: ignore[operator]
            if isinstance(op, ast.LtE):
                return a <= b  # type: ignore[operator]
            return None
        return None

    return ev(expr)


def _es_nodo_de_condicion(nodo: NodoCFG) -> Optional[ast.AST]:
    if isinstance(nodo.nodo_ast, ast.AnnAssign) and isinstance(nodo.nodo_ast.target, ast.Name):
        nombre = nodo.nodo_ast.target.id
        # Nota: el "_for" es una convencion interna para modelar el for como decision,
        # pero su "condicion" no significa lo mismo que en if/while, asi que no se evalua.
        if nombre in {"_if", "_while", "_elif"}:
            return nodo.nodo_ast.annotation
    return None


def marcar_partes_optimizables(inicio: NodoCFG, cache: Dict[int, NodoCFG]) -> Tuple[Set[Tuple[int, int]], Set[int]]:
    """Marca nodos que suelen ser candidatos a optimizacion automatica.

    Retorna:
        aristas_imposibles: aristas que caen por una condicion constante.
        inalcanzables: ids de nodos no alcanzables si se eliminan esas aristas.
    """
    aristas_imposibles: Set[Tuple[int, int]] = set()

    # 1) Condiciones constantes: if/while/for etiquetados
    for nodo in cache.values():
        expr = _es_nodo_de_condicion(nodo)
        if expr is None:
            continue

        valor = _evaluar_constante(expr)
        if isinstance(valor, bool):
            if valor:
                nodo.setdefault("marcas", set()).add("COND_CONST_TRUE")
            else:
                nodo.setdefault("marcas", set()).add("COND_CONST_FALSE")

            for hijo in getattr(nodo, "hijos", []):
                if valor is True and hijo.tipo_rama is False:
                    aristas_imposibles.add((nodo.id, hijo.id))
                if valor is False and hijo.tipo_rama is True:
                    aristas_imposibles.add((nodo.id, hijo.id))

    # 2) Asignacion muerta (muy simple): x=... seguido de x=... sin leer x
    for nodo in cache.values():
        if not isinstance(nodo.nodo_ast, ast.Assign):
            continue
        escritos = _nombres_escritos(nodo.nodo_ast)
        if len(escritos) != 1:
            continue
        var = next(iter(escritos))

        hijos = getattr(nodo, "hijos", [])
        if len(hijos) != 1:
            continue

        siguiente = hijos[0]
        if not isinstance(siguiente.nodo_ast, ast.Assign):
            continue

        escritos_sig = _nombres_escritos(siguiente.nodo_ast)
        if var not in escritos_sig:
            continue

        leidos_en_sig = _nombres_leidos(siguiente.nodo_ast.value)
        if var in leidos_en_sig:
            continue

        nodo.setdefault("marcas", set()).add("ASIGNACION_MUERTA")

    # 3) Alcanzabilidad ignorando aristas imposibles
    visitados: Set[int] = set()
    pila = [inicio.id]
    while pila:
        actual = pila.pop()
        if actual in visitados:
            continue
        visitados.add(actual)

        for hijo in cache[actual].hijos:
            if (actual, hijo.id) in aristas_imposibles:
                continue
            pila.append(hijo.id)

    inalcanzables = set(cache.keys()) - visitados
    for nid in inalcanzables:
        cache[nid].setdefault("marcas", set()).add("INALCANZABLE")

    return aristas_imposibles, inalcanzables


def _atributos_de_nodo(marcas: Set[str]) -> Dict[str, str]:
    """Define estilo visual de un nodo segun sus marcas."""
    attrs: Dict[str, str] = {}
    # Abuso controlado de ifs para que la grafica del propio programa tenga estructura.
    if "INALCANZABLE" in marcas:
        attrs["style"] = "filled"
        attrs["fillcolor"] = "gray90"
        if "COND_CONST_TRUE" in marcas or "COND_CONST_FALSE" in marcas:
            attrs["color"] = "gray50"
        else:
            attrs["color"] = "gray70"
    else:
        if "COND_CONST_TRUE" in marcas:
            attrs["style"] = "filled"
            attrs["fillcolor"] = "palegreen"
        else:
            if "COND_CONST_FALSE" in marcas:
                attrs["style"] = "filled"
                attrs["fillcolor"] = "mistyrose"
            else:
                if "ASIGNACION_MUERTA" in marcas:
                    attrs["style"] = "filled"
                    attrs["fillcolor"] = "lightgoldenrod"
                else:
                    # sin estilo especial
                    if False:  # rama intencionalmente imposible para la demo
                        attrs["fillcolor"] = "white"
    return attrs


def construir_grafo(cache: Dict[int, NodoCFG], aristas_imposibles: Set[Tuple[int, int]]) -> pygraphviz.AGraph:
    """Crea un grafo dirigido (DOT) desde los nodos del CFG, con resaltado basico."""

    grafo = pygraphviz.AGraph(directed=True)

    for _, nodo in cache.items():
        grafo.add_node(nodo.id)
        n = grafo.get_node(nodo.id)

        marcas = nodo.get("marcas", set())
        texto_fuente = _resumir_texto(_normalizar_etiqueta(nodo.fuente()))
        etiqueta = f"{nodo.linea()}: {texto_fuente}"
        if marcas:
            etiqueta = etiqueta + "\n[opt: " + ", ".join(sorted(marcas)) + "]"

        n.attr["label"] = etiqueta

        for clave, valor in _atributos_de_nodo(set(marcas)).items():
            n.attr[clave] = valor

        for padre in nodo.padres:
            if hasattr(padre, "enlace_llamada") and padre.enlace_llamada > 0 and not hasattr(nodo, "enlace_callee"):
                grafo.add_edge(padre.id, nodo.id, style="dotted", weight=100)
                continue

            attrs: Dict[str, str] = {}
            if (padre.id, nodo.id) in aristas_imposibles:
                attrs["style"] = "dashed"
                attrs["color"] = "gray50"
                attrs["label"] = "imposible"
            else:
                if nodo.tipo_rama is True:
                    attrs["color"] = "blue"
                    attrs["label"] = "T"
                else:
                    if nodo.tipo_rama is False:
                        attrs["color"] = "red"
                        attrs["label"] = "F"

            if attrs:
                grafo.add_edge(padre.id, nodo.id, **attrs)
            else:
                grafo.add_edge(padre.id, nodo.id)

    return grafo


def agregar_leyenda_si_aplica(grafo: pygraphviz.AGraph, cache: Dict[int, NodoCFG]) -> None:
    """Agrega una leyenda al grafo cuando existe algo que resaltar.

    La leyenda se coloca al final (rank sink) para que sea facil de leer.
    """

    hay_marcas = any(bool(n.get("marcas")) for n in cache.values())
    if not hay_marcas:
        return

    etiqueta = (
        '<<TABLE BORDER="0" CELLBORDER="1" CELLSPACING="0" CELLPADDING="4">'
        '<TR><TD COLSPAN="2"><B>Simbologia</B></TD></TR>'
        '<TR><TD BGCOLOR="palegreen"> </TD><TD>COND_CONST_TRUE: condicion siempre verdadera</TD></TR>'
        '<TR><TD BGCOLOR="mistyrose"> </TD><TD>COND_CONST_FALSE: condicion siempre falsa</TD></TR>'
        '<TR><TD BGCOLOR="lightgoldenrod"> </TD><TD>ASIGNACION_MUERTA: asignacion se pisa sin leerse</TD></TR>'
        '<TR><TD BGCOLOR="gray90"> </TD><TD>INALCANZABLE: no se llega al eliminar ramas imposibles</TD></TR>'
        '<TR><TD>azul / T</TD><TD>arista por rama verdadera</TD></TR>'
        '<TR><TD>rojo / F</TD><TD>arista por rama falsa</TD></TR>'
        '<TR><TD>dashed</TD><TD>arista imposible por condicion constante</TD></TR>'
        '</TABLE>>'
    )

    nombre = 'leyenda'
    grafo.add_node(nombre)
    nodo = grafo.get_node(nombre)
    nodo.attr['shape'] = 'plaintext'
    nodo.attr['label'] = etiqueta

    sub = grafo.add_subgraph([nombre], name='sub_leyenda')
    sub.graph_attr['rank'] = 'sink'

    # Conecta de forma invisible para estabilizar la disposicion.
    try:
        inicio = min(int(k) for k in cache.keys())
        grafo.add_edge(str(inicio), nombre, style='invis')
    except Exception:
        pass


def reiniciar_estado_global() -> None:
    """Resetea el estado compartido de `NodoCFG` para una ejecucion limpia."""

    NodoCFG.cache = {}
    NodoCFG.contador = 0


def generar_png_desde_archivo(ruta_python: Path) -> Path:
    """Genera el PNG del CFG a partir de un archivo Python."""

    codigo = ruta_python.read_text(encoding="utf-8")

    reiniciar_estado_global()
    generador = GeneradorCFG()
    generador.generar(codigo.strip())

    aristas_imposibles, _ = marcar_partes_optimizables(
        generador.nodo_inicio, NodoCFG.cache)

    grafo = construir_grafo(NodoCFG.cache, aristas_imposibles)

    agregar_leyenda_si_aplica(grafo, NodoCFG.cache)

    ruta_png = ruta_python.with_suffix(".png")
    grafo.draw(str(ruta_png), prog="dot")
    return ruta_png


def main(argv: List[str]) -> int:
    if len(argv) != 2:
        print("Uso: python3 cfg_generator.py ruta/al/programa.py", file=sys.stderr)
        return 2

    ruta = Path(argv[1])
    if not ruta.exists() or not ruta.is_file():
        print(f"Error: no se encontro el archivo: {ruta}", file=sys.stderr)
        return 2

    if ruta.suffix.lower() != ".py":
        print("Error: se espera un archivo con extension .py", file=sys.stderr)
        return 2

    try:
        salida = generar_png_desde_archivo(ruta)
    except (SyntaxError, UnicodeDecodeError) as exc:
        print(f"Error al parsear el archivo: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:  # pragma: no cover
        print(f"Error inesperado: {exc}", file=sys.stderr)
        return 1

    print(str(salida))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
