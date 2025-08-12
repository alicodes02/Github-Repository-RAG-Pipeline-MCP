import ast
import re
from typing import List, Dict, Any, Tuple

from langchain_text_splitters import (
    Language,
    RecursiveCharacterTextSplitter,
)

class PythonSemanticChunker:
    """Chunks Python code based on semantic units like functions, classes, etc."""
    
    def __init__(self, max_chunk_size: int = 2000, overlap_lines: int = 5):
        self.max_chunk_size = max_chunk_size
        self.overlap_lines = overlap_lines
    
    def extract_semantic_chunks(self, code: str, file_path: str) -> List[Dict[str, Any]]:
        """Extract semantic chunks from Python code."""
        try:
            tree = ast.parse(code)
            chunks = []
            lines = code.split('\n')
            
            # Get all function and class boundaries first
            boundaries = self._get_function_boundaries(code)
            
            # Extract chunks for each boundary
            for start_line, end_line in boundaries:
                # Find the corresponding AST node
                for node in ast.walk(tree):
                    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                        node_start = node.lineno - 1
                        if hasattr(node, 'decorator_list') and node.decorator_list:
                            node_start = node.decorator_list[0].lineno - 1
                        
                        if node_start == start_line:
                            chunk_info = self._create_chunk_from_boundaries(
                                node, lines, file_path, start_line, end_line
                            )
                            if chunk_info:
                                chunks.append(chunk_info)
                            break
            
            # Handle remaining code (imports, module-level variables, etc.)
            covered_lines = set()
            for start, end in boundaries:
                covered_lines.update(range(start, end + 1))
            
            remaining_chunks = self._extract_remaining_code(lines, covered_lines, file_path)
            chunks.extend(remaining_chunks)
            
            # Sort chunks by line number
            chunks.sort(key=lambda x: x['start_line'])
            
            # Split large chunks if necessary
            final_chunks = []
            for chunk in chunks:
                if len(chunk['text']) > self.max_chunk_size:
                    split_chunks = self._split_large_chunk(chunk)
                    final_chunks.extend(split_chunks)
                else:
                    final_chunks.append(chunk)
            
            return final_chunks
            
        except SyntaxError as e:
            # Fallback to line-based chunking for files with syntax errors
            return self._fallback_chunking(code, file_path, f"Syntax error: {e}")
    
    def _create_chunk_from_boundaries(self, node: ast.AST, lines: List[str], 
                                    file_path: str, start_line: int, end_line: int) -> Dict[str, Any]:
        """Create a chunk from pre-calculated boundaries."""
        # Extract the actual code
        chunk_lines = lines[start_line:end_line + 1]
        text = '\n'.join(chunk_lines)
        
        # Determine chunk type and name
        chunk_type = self._get_node_type(node)
        chunk_name = getattr(node, 'name', 'unknown')
        
        # Extract docstring if present
        docstring = ast.get_docstring(node) or ""
        
        # Extract function/method signatures
        signature = self._extract_signature(node, lines)
        
        return {
            'text': text,
            'start_line': start_line + 1,  # Convert back to 1-based for metadata
            'end_line': end_line + 1,
            'chunk_type': chunk_type,
            'chunk_name': chunk_name,
            'signature': signature,
            'docstring': docstring,
            'file_path': file_path,
            'line_count': end_line - start_line + 1
        }
    
    def _extract_node_chunk(self, node: ast.AST, lines: List[str], file_path: str) -> Dict[str, Any]:
        """Extract a chunk for a specific AST node (function or class)."""
        start_line = node.lineno - 1  # Convert to 0-based indexing
        
        # Include decorators if present
        if hasattr(node, 'decorator_list') and node.decorator_list:
            start_line = node.decorator_list[0].lineno - 1
        
        # Find the actual end of the function/class by analyzing indentation
        end_line = self._find_actual_end_line(node, lines, start_line)
        
        # Extract the actual code
        chunk_lines = lines[start_line:end_line + 1]
        text = '\n'.join(chunk_lines)
        
        # Determine chunk type and name
        chunk_type = self._get_node_type(node)
        chunk_name = getattr(node, 'name', 'unknown')
        
        # Extract docstring if present
        docstring = ast.get_docstring(node) or ""
        
        # Extract function/method signatures
        signature = self._extract_signature(node, lines)
        
        return {
            'text': text,
            'start_line': start_line + 1,  # Convert back to 1-based for metadata
            'end_line': end_line + 1,
            'chunk_type': chunk_type,
            'chunk_name': chunk_name,
            'signature': signature,
            'docstring': docstring,
            'file_path': file_path,
            'line_count': end_line - start_line + 1
        }
    
    def _find_actual_end_line(self, node: ast.AST, lines: List[str], start_line: int) -> int:
        """Find the actual end line of a function or class by analyzing indentation."""
        def_line_idx = node.lineno - 1  # The line with def/class keyword
        
        # Get the base indentation level (indentation of def/class line)
        def_line = lines[def_line_idx]
        base_indent = len(def_line) - len(def_line.lstrip())
        
        # Start checking from the line after the definition
        current_line = def_line_idx + 1
        end_line = def_line_idx  # Default to def line if nothing found
        
        # Look for the actual end by finding where indentation returns to base level or less
        while current_line < len(lines):
            line = lines[current_line]
            
            # Skip empty lines and comments
            stripped_line = line.strip()
            if not stripped_line or stripped_line.startswith('#'):
                current_line += 1
                continue
            
            # Calculate current indentation
            current_indent = len(line) - len(line.lstrip())
            
            # If we've returned to the base indentation level or less, we've found the end
            if current_indent <= base_indent:
                # Don't include this line as it's not part of our function/class
                end_line = current_line - 1
                break
            
            # This line is still part of our function/class
            end_line = current_line
            current_line += 1
        
        # Handle case where function/class goes to end of file
        if current_line >= len(lines):
            end_line = len(lines) - 1
        
        return end_line
    
    def _get_function_boundaries(self, code: str) -> List[Tuple[int, int]]:
        """Get all function and class boundaries in the code."""
        try:
            tree = ast.parse(code)
            boundaries = []
            lines = code.split('\n')
            
            for node in ast.walk(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                    start_line = node.lineno - 1
                    
                    # Include decorators if present
                    if hasattr(node, 'decorator_list') and node.decorator_list:
                        start_line = node.decorator_list[0].lineno - 1
                    
                    end_line = self._find_actual_end_line(node, lines, start_line)
                    boundaries.append((start_line, end_line))
            
            # Sort by start line
            boundaries.sort()
            return boundaries
            
        except SyntaxError:
            return []
        """Get the type of AST node."""
        if isinstance(node, ast.ClassDef):
            return 'class'
        elif isinstance(node, ast.FunctionDef):
            return 'function'
        elif isinstance(node, ast.AsyncFunctionDef):
            return 'async_function'
        return 'unknown'
    
    def _extract_signature(self, node: ast.AST, lines: List[str]) -> str:
        """Extract function/class signature."""
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            # Get the function definition line
            start_line = node.lineno - 1
            
            # Handle decorators
            if hasattr(node, 'decorator_list') and node.decorator_list:
                decorator_lines = []
                for decorator in node.decorator_list:
                    decorator_lines.append(lines[decorator.lineno - 1].strip())
                decorators = '\n'.join(decorator_lines)
                return f"{decorators}\n{lines[start_line].strip()}"
            
            return lines[start_line].strip()
        
        elif isinstance(node, ast.ClassDef):
            start_line = node.lineno - 1
            
            # Handle decorators for classes too
            if hasattr(node, 'decorator_list') and node.decorator_list:
                decorator_lines = []
                for decorator in node.decorator_list:
                    decorator_lines.append(lines[decorator.lineno - 1].strip())
                decorators = '\n'.join(decorator_lines)
                return f"{decorators}\n{lines[start_line].strip()}"
            
            return lines[start_line].strip()
        
        return ""
    
    def _extract_remaining_code(self, lines: List[str], covered_lines: set, file_path: str) -> List[Dict[str, Any]]:
        """Extract chunks for code not covered by functions/classes."""
        remaining_chunks = []
        current_chunk_lines = []
        current_start_line = None
        
        for i, line in enumerate(lines):
            line_num = i + 1
            
            if line_num not in covered_lines:
                if current_start_line is None:
                    current_start_line = line_num
                current_chunk_lines.append(line)
            else:
                # We hit a covered line, finalize current chunk if it exists
                if current_chunk_lines:
                    chunk_text = '\n'.join(current_chunk_lines).strip()
                    if chunk_text:  # Only add non-empty chunks
                        chunk_type = self._determine_remaining_chunk_type(chunk_text)
                        remaining_chunks.append({
                            'text': chunk_text,
                            'start_line': current_start_line,
                            'end_line': current_start_line + len(current_chunk_lines) - 1,
                            'chunk_type': chunk_type,
                            'chunk_name': f"{chunk_type}_block",
                            'signature': "",
                            'docstring': "",
                            'file_path': file_path,
                            'line_count': len(current_chunk_lines)
                        })
                    current_chunk_lines = []
                    current_start_line = None
        
        # Handle any remaining lines at the end
        if current_chunk_lines:
            chunk_text = '\n'.join(current_chunk_lines).strip()
            if chunk_text:
                chunk_type = self._determine_remaining_chunk_type(chunk_text)
                remaining_chunks.append({
                    'text': chunk_text,
                    'start_line': current_start_line,
                    'end_line': current_start_line + len(current_chunk_lines) - 1,
                    'chunk_type': chunk_type,
                    'chunk_name': f"{chunk_type}_block",
                    'signature': "",
                    'docstring': "",
                    'file_path': file_path,
                    'line_count': len(current_chunk_lines)
                })
        
        return remaining_chunks
    
    def _determine_remaining_chunk_type(self, text: str) -> str:
        """Determine the type of remaining code chunk."""
        text_stripped = text.strip()
        
        if re.match(r'^(import|from)\s+', text_stripped):
            return 'imports'
        elif re.search(r'^[A-Z_][A-Z0-9_]*\s*=', text_stripped):
            return 'constants'
        elif re.search(r'^[a-zA-Z_][a-zA-Z0-9_]*\s*=', text_stripped):
            return 'variables'
        elif text_stripped.startswith('"""') or text_stripped.startswith("'''"):
            return 'module_docstring'
        elif text_stripped.startswith('#'):
            return 'comments'
        else:
            return 'module_level_code'
    
    def _split_large_chunk(self, chunk: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Split large chunks using traditional text splitting as fallback."""
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=self.max_chunk_size,
            chunk_overlap=200,
            separators=["\n\n", "\n", " ", ""]
        )
        
        texts = splitter.split_text(chunk['text'])
        chunks = []
        
        for i, text in enumerate(texts):
            new_chunk = chunk.copy()
            new_chunk['text'] = text
            new_chunk['chunk_name'] = f"{chunk['chunk_name']}_part_{i+1}"
            new_chunk['is_split'] = True
            chunks.append(new_chunk)
        
        return chunks
    
    def _fallback_chunking(self, code: str, file_path: str, error_msg: str) -> List[Dict[str, Any]]:
        """Fallback to traditional chunking when AST parsing fails."""
        splitter = RecursiveCharacterTextSplitter.from_language(
            language=Language.PYTHON, 
            chunk_size=self.max_chunk_size, 
            chunk_overlap=200
        )
        
        chunks = splitter.split_text(code)
        result = []
        
        for i, chunk in enumerate(chunks):
            result.append({
                'text': chunk,
                'start_line': 1,  # Can't determine exact lines
                'end_line': len(chunk.split('\n')),
                'chunk_type': 'fallback',
                'chunk_name': f'fallback_chunk_{i+1}',
                'signature': "",
                'docstring': "",
                'file_path': file_path,
                'line_count': len(chunk.split('\n')),
                'error': error_msg
            })
        
        return result
    
    def _get_node_type(self, node: ast.AST) -> str:
        if isinstance(node, ast.ClassDef):
            return 'class'
        elif isinstance(node, ast.FunctionDef):
            return 'function'
        elif isinstance(node, ast.AsyncFunctionDef):
            return 'async_function'
        return 'unknown'
