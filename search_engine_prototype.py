"""
Search Engine Prototype
-----------------------
Core Data Structures:
 - Trie: for autocomplete of query terms
 - Inverted Index: maps token -> {doc_id: term_frequency}
 - AVL Tree: used to rank search results by score (higher score = higher rank)

Features:
 - Load a corpus of plain-text files from a directory ("mini Wikipedia")
 - Tokenize and normalize text, build inverted index and document statistics
 - Trie-based autocomplete for query suggestions
 - Query processing that computes a simple TF-IDF-like score for documents
 - Rank results using an AVL tree keyed by score
 - Return highlighted snippets showing matched terms
 - Save/load index to/from disk for persistence
 - Simple CLI for interacting with the prototype

This is an educational prototype (not production-ready). The implementation
prioritizes clarity and pedagogy over extreme performance. Still, it is
feature-rich enough to be used as a final-year project demo.

Usage:
  python search_engine_prototype.py

Author: Generated for student project (Advanced Data Structures)
Date: 2025-10-03
"""

import os
import re
import sys
import json
import math
import pickle
import argparse
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Any
from collections import defaultdict, Counter
import threading

# ----------------------------- Utilities ---------------------------------

RE_WORD = re.compile(r"\b[0-9A-Za-z']+\b")

ANSI_HL = '\x1b[93m'  # yellow
ANSI_RESET = '\x1b[0m'


def tokenize(text: str) -> List[str]:
    """Simple tokenizer: return list of lowercase tokens (keeping apostrophes)."""
    return [t.lower() for t in RE_WORD.findall(text)]


def make_snippet(text: str, terms: List[str], radius: int = 40) -> str:
    """Return a snippet with the first occurrence of any term highlighted."""
    low = text.lower()
    first_pos = None
    first_term = None
    for term in terms:
        p = low.find(term.lower())
        if p >= 0 and (first_pos is None or p < first_pos):
            first_pos = p
            first_term = term
    if first_pos is None:
        # return start of text
        s = text[:radius * 2]
        if len(text) > len(s):
            s += '...'
        return s
    start = max(0, first_pos - radius)
    end = min(len(text), first_pos + radius)
    prefix = '...' if start > 0 else ''
    suffix = '...' if end < len(text) else ''
    snippet = text[start:end]
    # highlight all terms in snippet (case-insensitive)
    def _repl(m):
        return f"{ANSI_HL}{m.group(0)}{ANSI_RESET}"
    # build pattern of terms
    pattern = re.compile('|'.join(re.escape(t) for t in set(terms)), re.IGNORECASE)
    snippet = pattern.sub(_repl, snippet)
    return prefix + snippet + suffix


# ----------------------------- Trie --------------------------------------

class TrieNode:
    __slots__ = ('children', 'is_end', 'freq')

    def __init__(self):
        self.children: Dict[str, 'TrieNode'] = {}
        self.is_end: bool = False
        self.freq: int = 0


class Trie:
    """Trie supporting insertion and prefix autocomplete."""
    def __init__(self):
        self.root = TrieNode()
        self.lock = threading.RLock()

    def insert(self, word: str) -> None:
        with self.lock:
            node = self.root
            for ch in word:
                if ch not in node.children:
                    node.children[ch] = TrieNode()
                node = node.children[ch]
            node.is_end = True
            node.freq += 1

    def _collect(self, node: TrieNode, prefix: str, collector: List[Tuple[str, int]]):
        if node.is_end:
            collector.append((prefix, node.freq))
        for ch, child in node.children.items():
            self._collect(child, prefix + ch, collector)

    def autocomplete(self, prefix: str, limit: int = 10) -> List[Tuple[str, int]]:
        with self.lock:
            node = self.root
            for ch in prefix:
                if ch not in node.children:
                    return []
                node = node.children[ch]
            res: List[Tuple[str, int]] = []
            self._collect(node, prefix, res)
            res.sort(key=lambda x: (-x[1], x[0]))
            return res[:limit]

    def to_dict(self) -> Dict[str, Any]:
        def node_to_dict(n: TrieNode):
            return {'is_end': n.is_end, 'freq': n.freq, 'children': {ch: node_to_dict(c) for ch, c in n.children.items()}}
        return node_to_dict(self.root)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> 'Trie':
        t = cls()
        def dict_to_node(nd: Dict[str, Any]) -> TrieNode:
            n = TrieNode()
            n.is_end = nd.get('is_end', False)
            n.freq = int(nd.get('freq', 0))
            for ch, chdict in nd.get('children', {}).items():
                n.children[ch] = dict_to_node(chdict)
            return n
        t.root = dict_to_node(d)
        return t


# ----------------------------- Inverted Index ----------------------------

@dataclass
class Document:
    id: int
    path: str
    title: str
    length: int
    tokens: List[str]


class InvertedIndex:
    def __init__(self):
        # token -> {doc_id: frequency}
        self.index: Dict[str, Dict[int, int]] = defaultdict(dict)
        # doc_id -> Document
        self.documents: Dict[int, Document] = {}
        self.doc_count = 0
        self.lock = threading.RLock()

    def add_document(self, doc_path: str, doc_id: Optional[int] = None) -> int:
        """Read file, tokenize, add to index. Returns doc_id used."""
        with self.lock:
            if doc_id is None:
                doc_id = self.doc_count + 1
            text = open(doc_path, 'r', encoding='utf-8', errors='ignore').read()
            tokens = tokenize(text)
            title = os.path.basename(doc_path)
            doc = Document(id=doc_id, path=doc_path, title=title, length=len(tokens), tokens=tokens)
            self.documents[doc_id] = doc
            self.doc_count = max(self.doc_count, doc_id)
            freqs = Counter(tokens)
            for tok, f in freqs.items():
                self.index[tok][doc_id] = f
            return doc_id

    def get_postings(self, token: str) -> Dict[int, int]:
        return dict(self.index.get(token, {}))

    def docs_with_token(self, token: str) -> List[int]:
        return list(self.index.get(token, {}).keys())

    def idf(self, token: str) -> float:
        df = len(self.index.get(token, {}))
        if df == 0:
            return 0.0
        return math.log((1 + self.doc_count) / (1 + df)) + 1.0

    def tf(self, token: str, doc_id: int) -> float:
        return float(self.index.get(token, {}).get(doc_id, 0))

    def save(self, filename: str):
        with open(filename, 'wb') as f:
            pickle.dump({'index': self.index, 'documents': self.documents, 'doc_count': self.doc_count}, f)

    @classmethod
    def load(cls, filename: str) -> 'InvertedIndex':
        with open(filename, 'rb') as f:
            payload = pickle.load(f)
        ii = cls()
        ii.index = payload['index']
        ii.documents = payload['documents']
        ii.doc_count = int(payload.get('doc_count', len(ii.documents)))
        return ii


# ----------------------------- AVL Tree ----------------------------------

class AVLNode:
    def __init__(self, key: float, doc_ids: Optional[List[int]] = None):
        self.key = key
        self.doc_ids = doc_ids or []
        self.left: Optional['AVLNode'] = None
        self.right: Optional['AVLNode'] = None
        self.height: int = 1


class AVLTree:
    """AVL tree keyed by numeric score. Each node stores list of document ids with that score."""
    def __init__(self):
        self.root: Optional[AVLNode] = None
        self.lock = threading.RLock()

    def _height(self, node: Optional[AVLNode]) -> int:
        return node.height if node else 0

    def _update_height(self, node: AVLNode) -> None:
        node.height = 1 + max(self._height(node.left), self._height(node.right))

    def _balance_factor(self, node: AVLNode) -> int:
        return self._height(node.left) - self._height(node.right)

    def _rotate_right(self, y: AVLNode) -> AVLNode:
        x = y.left
        T2 = x.right
        x.right = y
        y.left = T2
        self._update_height(y)
        self._update_height(x)
        return x

    def _rotate_left(self, x: AVLNode) -> AVLNode:
        y = x.right
        T2 = y.left
        y.left = x
        x.right = T2
        self._update_height(x)
        self._update_height(y)
        return y

    def _rebalance(self, node: AVLNode) -> AVLNode:
        self._update_height(node)
        bf = self._balance_factor(node)
        if bf > 1:
            if self._balance_factor(node.left) < 0:
                node.left = self._rotate_left(node.left)
            return self._rotate_right(node)
        if bf < -1:
            if self._balance_factor(node.right) > 0:
                node.right = self._rotate_right(node.right)
            return self._rotate_left(node)
        return node

    def _insert_node(self, node: Optional[AVLNode], key: float, doc_id: int) -> AVLNode:
        if node is None:
            return AVLNode(key, [doc_id])
        if abs(key - node.key) < 1e-9:
            if doc_id not in node.doc_ids:
                node.doc_ids.append(doc_id)
            return node
        elif key < node.key:
            node.left = self._insert_node(node.left, key, doc_id)
        else:
            node.right = self._insert_node(node.right, key, doc_id)
        return self._rebalance(node)

    def insert(self, key: float, doc_id: int) -> None:
        with self.lock:
            self.root = self._insert_node(self.root, key, doc_id)

    def _inorder_desc(self, node: Optional[AVLNode], collector: List[Tuple[float, List[int]]], limit: Optional[int]):
        if node is None:
            return
        if limit is not None and len(collector) >= limit:
            return
        # traverse right (higher keys) first for descending order
        self._inorder_desc(node.right, collector, limit)
        if limit is not None and len(collector) >= limit:
            return
        collector.append((node.key, list(node.doc_ids)))
        if limit is not None and len(collector) >= limit:
            return
        self._inorder_desc(node.left, collector, limit)

    def top_k(self, k: int = 10) -> List[Tuple[float, List[int]]]:
        res: List[Tuple[float, List[int]]] = []
        with self.lock:
            self._inorder_desc(self.root, res, k)
        return res


# ----------------------------- Search Engine -----------------------------

class SearchEngine:
    def __init__(self):
        self.trie = Trie()
        self.index = InvertedIndex()
        # precomputed document vector lengths for cosine-like normalization
        self.doc_norms: Dict[int, float] = {}
        self.lock = threading.RLock()

    def build_from_folder(self, folder: str) -> None:
        """Walk folder and index all .txt files (non-recursive by default)."""
        files = [os.path.join(folder, f) for f in os.listdir(folder) if f.endswith('.txt')]
        if not files:
            print('No .txt files found in', folder)
            return
        print(f'Indexing {len(files)} files...')
        for i, path in enumerate(files, start=1):
            try:
                doc_id = self.index.add_document(path)
                # insert tokens into trie for autocomplete
                toks = set(self.index.documents[doc_id].tokens)
                for t in toks:
                    self.trie.insert(t)
            except Exception as ex:
                print('Failed to index', path, ex)
        # compute norms
        self._compute_doc_norms()
        print('Index built. Documents:', len(self.index.documents))

    def _compute_doc_norms(self) -> None:
        # compute vector length for each document using tf-idf weights
        self.doc_norms = {}
        for doc_id, doc in self.index.documents.items():
            s = 0.0
            freqs = Counter(doc.tokens)
            for tok, tf in freqs.items():
                idf = self.index.idf(tok)
                w = (1 + math.log(tf)) * idf if tf > 0 else 0
                s += w * w
            self.doc_norms[doc_id] = math.sqrt(s)

    def save(self, filename: str) -> None:
        payload = {
            'trie': self.trie.to_dict(),
        }
        with open(filename + '.meta', 'wb') as f:
            pickle.dump(payload, f)
        self.index.save(filename + '.idx')
        with open(filename + '.norms', 'wb') as f:
            pickle.dump(self.doc_norms, f)
        print('saved index to', filename + '.*')

    def load(self, filename: str) -> None:
        with open(filename + '.meta', 'rb') as f:
            payload = pickle.load(f)
        self.trie = Trie.from_dict(payload['trie'])
        self.index = InvertedIndex.load(filename + '.idx')
        with open(filename + '.norms', 'rb') as f:
            self.doc_norms = pickle.load(f)
        print('loaded index from', filename + '.*')

    def autocomplete(self, prefix: str, limit: int = 10) -> List[str]:
        res = self.trie.autocomplete(prefix, limit=limit)
        return [w for w, _ in res]

    def _score_document(self, query_terms: List[str], doc_id: int) -> float:
        # compute simple cosine similarity between query and doc using tf-idf weights
        # query vector weights: use term frequency in query
        qfreq = Counter(query_terms)
        num = 0.0
        qnorm = 0.0
        for term, qtf in qfreq.items():
            idf = self.index.idf(term)
            wq = (1 + math.log(qtf)) * idf if qtf > 0 else 0
            qnorm += wq * wq
            tf = self.index.tf(term, doc_id)
            wd = (1 + math.log(tf)) * idf if tf > 0 else 0
            num += wq * wd
        qnorm = math.sqrt(qnorm) if qnorm > 0 else 1.0
        dnorm = self.doc_norms.get(doc_id, 1.0)
        if dnorm == 0:
            return 0.0
        return num / (qnorm * dnorm)

    def search(self, query: str, top_k: int = 10) -> List[Tuple[float, int]]:
        terms = tokenize(query)
        if not terms:
            return []
        # get candidate docs: union of postings lists
        candidate_set = set()
        for t in terms:
            postings = self.index.docs_with_token(t)
            candidate_set.update(postings)
        # score candidates using AVL tree
        avl = AVLTree()
        for doc_id in candidate_set:
            score = self._score_document(terms, doc_id)
            if score > 0:
                avl.insert(score, doc_id)
        ranked = avl.top_k(top_k)
        # flatten into list of (score, doc_id)
        out: List[Tuple[float, int]] = []
        for score, doc_ids in ranked:
            for did in doc_ids:
                out.append((score, did))
                if len(out) >= top_k:
                    return out
        return out


# ----------------------------- CLI --------------------------------------

def print_result(engine: SearchEngine, results: List[Tuple[float, int]]):
    if not results:
        print('No results')
        return
    for rank, (score, doc_id) in enumerate(results, start=1):
        doc = engine.index.documents[doc_id]
        print(f"{rank:2d}. [{score:.4f}] {doc.title} (id={doc_id}, tokens={doc.length})")
        text = open(doc.path, 'r', encoding='utf-8', errors='ignore').read()
        snippet = make_snippet(text, [], radius=120)  # no highlights by default
        print('    ', snippet)


def interactive_shell(engine: SearchEngine):
    menu = """
Search Engine Prototype Menu:
 1) Build index from folder
 2) Save index
 3) Load index
 4) Autocomplete suggestions
 5) Search query
 6) Show document by id
 7) Export doc text (plain)
 0) Exit
"""
    while True:
        print(menu)
        choice = input('choice> ').strip()
        if choice == '0':
            return
        elif choice == '1':
            folder = input('folder path (contains .txt files)> ').strip()
            if not folder:
                print('cancel')
                continue
            engine.build_from_folder(folder)
        elif choice == '2':
            filename = input('base filename to save (e.g. myindex)> ').strip()
            if not filename:
                print('cancel')
                continue
            engine.save(filename)
        elif choice == '3':
            filename = input('base filename to load (e.g. myindex)> ').strip()
            if not filename:
                print('cancel')
                continue
            engine.load(filename)
        elif choice == '4':
            prefix = input('prefix> ').strip()
            res = engine.autocomplete(prefix, limit=20)
            print('Suggestions:')
            for i, s in enumerate(res, start=1):
                print(f" {i}. {s}")
        elif choice == '5':
            q = input('query> ').strip()
            topk = input('top k (default 10)> ').strip() or '10'
            try:
                k = int(topk)
            except:
                k = 10
            results = engine.search(q, top_k=k)
            # print with highlighted snippets
            if not results:
                print('No results')
                continue
            for rank, (score, doc_id) in enumerate(results, start=1):
                doc = engine.index.documents[doc_id]
                text = open(doc.path, 'r', encoding='utf-8', errors='ignore').read()
                snippet = make_snippet(text, tokenize(q), radius=120)
                print(f"{rank:2d}. [{score:.4f}] {doc.title} (id={doc_id})")
                print('    ', snippet)
        elif choice == '6':
            did = input('doc id> ').strip()
            try:
                didi = int(did)
                doc = engine.index.documents.get(didi)
                if not doc:
                    print('not found')
                    continue
                print('Title:', doc.title)
                print('Path:', doc.path)
                print('Tokens:', doc.length)
                print('\n----CONTENT----\n')
                print(open(doc.path, 'r', encoding='utf-8', errors='ignore').read())
            except Exception as ex:
                print('error:', ex)
        elif choice == '7':
            did = input('doc id> ').strip()
            outname = input('output filename> ').strip() or None
            try:
                didi = int(did)
                doc = engine.index.documents.get(didi)
                if not doc:
                    print('not found')
                    continue
                text = open(doc.path, 'r', encoding='utf-8', errors='ignore').read()
                if outname is None:
                    outname = f"doc_{didi}.txt"
                with open(outname, 'w', encoding='utf-8') as f:
                    f.write(text)
                print('exported to', outname)
            except Exception as ex:
                print('error:', ex)
        else:
            print('unknown choice')


# ----------------------------- Demo -------------------------------------

def generate_sample_corpus(folder: str, n: int = 20):
    """Create a small sample corpus of text files inside `folder` for demo/testing."""
    os.makedirs(folder, exist_ok=True)
    lorem = (
        "Lorem ipsum dolor sit amet, consectetur adipiscing elit. Sed non risus. Suspendisse lectus tortor, "
        "dignissim sit amet, adipiscing nec, ultricies sed, dolor. Cras elementum ultrices diam. Maecenas ligula massa, "
        "varius a, semper congue, euismod non, mi. Proin porttitor, orci nec nonummy molestie, enim est eleifend mi, "
        "non fermentum diam nisl sit amet erat. Duis semper. Duis arcu massa, scelerisque vitae, consequat in, pretium a, enim."
    )
    for i in range(1, n + 1):
        fname = os.path.join(folder, f'doc_{i:03d}.txt')
        with open(fname, 'w', encoding='utf-8') as f:
            # vary content
            content = f"Document {i}\n\n{lorem}\n\nMore content about data structures, trees, trie, inverted index, avl, search, algorithms.\n"
            f.write(content)
    print('sample corpus generated in', folder)


# ----------------------------- Main -------------------------------------

def main(argv):
    parser = argparse.ArgumentParser(description='Search Engine Prototype CLI')
    parser.add_argument('--demo', action='store_true', help='generate demo corpus and build index')
    parser.add_argument('--corpus', type=str, default='corpus', help='corpus folder containing .txt')
    args = parser.parse_args(argv[1:])

    engine = SearchEngine()
    if args.demo:
        generate_sample_corpus(args.corpus, n=30)
        engine.build_from_folder(args.corpus)
        print('Demo index ready. Launching interactive shell...')
    interactive_shell(engine)


if __name__ == '__main__':
    main(sys.argv)
