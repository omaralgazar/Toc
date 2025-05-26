from typing import Any
import graphviz
from collections import defaultdict
import networkx as nx
from graphviz import Digraph, dot
from matplotlib import pyplot as plt, patches


class State:
    def __init__(self):
        self.transitions = defaultdict(list)
        self.epsilon = []


class NFA:
    def __init__(self, start, accept):
        self.start = start
        self.accept = accept


def regex_to_nfa(regex):
    def parse():
        nonlocal i
        result = term()
        while i < len(regex) and regex[i] == '|':
            i += 1
            result = alternate(result, term())
        return result

    def term():
        nonlocal i
        result = factor()
        while i < len(regex) and regex[i] not in '|)':
            result = concatenate(result, factor())
        return result

    def factor():
        nonlocal i
        base = None
        if regex[i] == '(':
            i += 1
            base = parse()
            i += 1
        else:
            base = character(regex[i])
            i += 1

        while i < len(regex) and regex[i] in '*+':
            if regex[i] == '*':
                base = kleene_star(base)
            elif regex[i] == '+':
                base = plus(base)
            i += 1

        return base

    def character(c):
        s1, s2 = State(), State()
        s1.transitions[c].append(s2)
        return NFA(s1, s2)

    def concatenate(nfa1, nfa2):
        nfa1.accept.epsilon.append(nfa2.start)
        return NFA(nfa1.start, nfa2.accept)

    def alternate(nfa1, nfa2):
        start, accept = State(), State()
        start.epsilon.extend([nfa1.start, nfa2.start])
        nfa1.accept.epsilon.append(accept)
        nfa2.accept.epsilon.append(accept)
        return NFA(start, accept)

    def kleene_star(nfa):
        start, accept = State(), State()
        start.epsilon.extend([nfa.start, accept])
        nfa.accept.epsilon.extend([nfa.start, accept])
        return NFA(start, accept)

    def plus(nfa):
        # a+ = a a*
        return concatenate(nfa, kleene_star(nfa))

    i = 0
    return parse()


class DFA:
    def __init__(self):
        self.states = []
        self.transitions = {}
        self.start_state = None
        self.accept_states = set()
        self.dead_state = None
        self.alphabet = set()


def nfa_to_dfa(nfa):
    def epsilon_closure(states):
        stack = list(states)
        closure = set(states)
        while stack:
            state = stack.pop()
            for next_state in state.epsilon:
                if next_state not in closure:
                    closure.add(next_state)
                    stack.append(next_state)
        return closure

    def move(states, symbol):
        result = set()
        for state in states:
            result.update(state.transitions.get(symbol, []))
        return result

    state_map = {}
    dfa = DFA()
    start_closure = frozenset(epsilon_closure([nfa.start]))
    state_queue = [start_closure]
    state_map[start_closure] = 0
    dfa.states.append(start_closure)
    dfa.start_state = 0

    all_symbols = set()

    while state_queue:
        current = state_queue.pop(0)
        current_id = state_map[current]
        symbols = set()
        for state in current:
            symbols.update(state.transitions.keys())
            all_symbols.update(state.transitions.keys())

        for symbol in symbols:
            if symbol == '': continue
            next_states = epsilon_closure(move(current, symbol))
            next_frozen = frozenset(next_states)
            if next_frozen not in state_map:
                state_map[next_frozen] = len(dfa.states)
                dfa.states.append(next_frozen)
                state_queue.append(next_frozen)
            dfa.transitions[(current_id, symbol)] = state_map[next_frozen]


    dfa.alphabet = all_symbols

    for states, sid in state_map.items():
        if nfa.accept in states:
            dfa.accept_states.add(sid)

    return dfa


def add_dead_state(dfa):

    dead_state_id = len(dfa.states)
    dfa.dead_state = dead_state_id
    dfa.states.append(frozenset())

    for state_id in range(len(dfa.states) - 1):
        for symbol in dfa.alphabet:
            if (state_id, symbol) not in dfa.transitions:
                dfa.transitions[(state_id, symbol)] = dead_state_id

    for symbol in dfa.alphabet:
        dfa.transitions[(dead_state_id, symbol)] = dead_state_id

    print(f"Dead state added: q{dead_state_id}")
    return dfa


def draw_dfa_with_arrows_and_accept(dfa):
    G = nx.DiGraph()
    state_name_map = {i: f"q{i}" for i in range(len(dfa.states))}

    for i in range(len(dfa.states)):
        name = state_name_map[i]
        is_dead = (i == dfa.dead_state)
        G.add_node(name,
                   is_accept=(i in dfa.accept_states),
                   is_start=(i == dfa.start_state),
                   is_dead=is_dead)

    edge_labels = {}
    for (from_state, symbol), to_state in dfa.transitions.items():
        from_name = state_name_map[from_state]
        to_name = state_name_map[to_state]
        G.add_edge(from_name, to_name)
        if (from_name, to_name) in edge_labels:
            edge_labels[(from_name, to_name)] += f",{symbol}"
        else:
            edge_labels[(from_name, to_name)] = symbol


    pos = nx.spring_layout(G, seed=42, scale=2, k=1.0, iterations=100)

    fig, ax = plt.subplots(figsize=(14, 10))

    node_colors = []
    for n in G.nodes():
        if G.nodes[n]['is_dead']:
            node_colors.append("lightcoral")
        elif G.nodes[n]['is_accept']:
            node_colors.append("lightgreen")
        elif G.nodes[n]['is_start']:
            node_colors.append("lightblue")
        else:
            node_colors.append("lightgray")

    nx.draw_networkx_nodes(G, pos, ax=ax, node_size=3000,
                           node_color=node_colors, edgecolors="black", linewidths=1.5)

    for (from_name, to_name), label in edge_labels.items():
        x1, y1 = pos[from_name]
        x2, y2 = pos[to_name]
        mid_x, mid_y = (x1 + x2) / 2, (y1 + y2) / 2

        arrow = patches.FancyArrow(mid_x, mid_y, (x2 - x1) * 0.1, (y2 - y1) * 0.1,
                                   width=0.02, head_width=0.15, head_length=0.1,
                                   color="red", alpha=0.8)
        ax.add_patch(arrow)

    labels = {}
    for n in G.nodes():
        if G.nodes[n]['is_dead']:
            labels[n] = f"{n}\n(DEAD)"
        else:
            labels[n] = n

    nx.draw_networkx_labels(G, pos, labels=labels, font_size=12, font_weight="bold", ax=ax)

    nx.draw_networkx_edges(G, pos, ax=ax,
                           arrowstyle="-|>", arrowsize=20,
                           edge_color="black", width=2,
                           connectionstyle="arc3,rad=0.15")

    nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels,
                                 font_size=10, font_color="dimgray", ax=ax)

    start_name = state_name_map[dfa.start_state]
    x, y = pos[start_name]
    ax.annotate("",
                xy=(x, y), xycoords='data',
                xytext=(x - 0.7, y), textcoords='data',
                arrowprops=dict(arrowstyle="->", color="green", lw=3),
                )
    ax.text(x - 0.85, y + 0.1, "Start", fontsize=12, color="green", weight="bold")

    for i, node in enumerate(G.nodes()):
        if G.nodes[node]['is_accept']:
            x, y = pos[node]
            circle = plt.Circle((x, y), 0.17, fill=False, color='black', linewidth=2)
            ax.add_patch(circle)

    ax.set_aspect('equal')
    plt.title("DFA with Dead State", fontsize=16, weight="bold")

    legend_elements = [
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightblue',
                   markersize=10, label='Start State'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgreen',
                   markersize=10, label='Accept State'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightcoral',
                   markersize=10, label='Dead State'),
        plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='lightgray',
                   markersize=10, label='Regular State')
    ]
    ax.legend(handles=legend_elements, loc='upper right')

    plt.axis("off")
    plt.tight_layout()
    plt.show()


def print_tm_from_dfa(dfa):
    print("\nTuring Machine from DFA:")
    print("=" * 50)

    states = [f"q{i}" for i in range(len(dfa.states))]

    alphabet = sorted(list(dfa.alphabet))

    tape_alphabet = alphabet + ['#', 'Y', 'N']

    print(f"States (K): {states + ['qAccept', 'qReject']}")
    print(f"Input Alphabet (Σ): {alphabet}")
    print(f"Tape Alphabet (Γ): {tape_alphabet}")
    print(f"Start State (S): q{dfa.start_state}")
    print(f"Accept States in DFA: {[f'q{s}' for s in dfa.accept_states]}")
    if dfa.dead_state is not None:
        print(f"Dead State: q{dfa.dead_state}")
    print()
    print("Transition Function (δ):")
    print("-" * 30)


    for (from_state, symbol), to_state in sorted(dfa.transitions.items()):
        transition_note = ""
        if to_state == dfa.dead_state:
            transition_note = "  // Transition to Dead State"
        elif from_state == dfa.dead_state:
            transition_note = "  // Dead State Loop"
        print(f"δ(q{from_state}, {symbol}) = (q{to_state}, {symbol}, R){transition_note}")

    print()

    for i in range(len(dfa.states)):
        if i in dfa.accept_states:
            print(f"δ(q{i}, #) = (q{i}, Y, L) ")
        else:

            print(f"δ(q{i}, #) = (q{i}, N, L) ")

    print()
    print("Final States:")
    print("qAccept: Accepting (Halting) state")
    print("qReject: Rejecting (Halting) state")
    if dfa.dead_state is not None:
        print(f"q{dfa.dead_state}: Dead state (trap state) - all undefined transitions lead here")



if __name__ == "__main__":
    regex = input("Enter Regular Expression: ").replace(" ", "")
    print(f"Regular Expression: {regex}")

    nfa = regex_to_nfa(regex)
    dfa = nfa_to_dfa(nfa)
    dfa = add_dead_state(dfa)
    draw_dfa_with_arrows_and_accept(dfa)
    print_tm_from_dfa(dfa)