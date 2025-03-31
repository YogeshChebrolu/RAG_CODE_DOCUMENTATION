import re
from typing import Tuple, List
from langchain_core.documents import Document

# Global chunk_id tracker
global_chunk_id = 0  # Ensures unique chunk_id across multiple calls

def chunk_text(text: str, url: str = "https://pydantic.com", chunk_size: int = 2000) -> Tuple[List[Document], List[Document]]:
    """
    Splits markdown content into linked text/code chunks while preserving original structure.
    - Text and code chunks share the same chunk_id when they appear together in the source
    - Maintains original order of both text and code blocks
    - Includes improved boundary detection for natural text splitting
    """
    global global_chunk_id  # Use global variable to keep chunk_id unique
    
    # Improved code block detection with optional language specifier
    code_block_re = re.compile(r"```(?:[a-zA-Z]+)?\n(.*?)```", re.DOTALL)
    code_blocks = code_block_re.findall(text)
    
    # Create text with code block placeholders while preserving positions
    text_with_placeholders = code_block_re.sub("[CODE_BLOCK]", text)
    
    # Split text into chunks with natural boundaries
    text_chunks = []
    start = 0
    text_length = len(text_with_placeholders)
    
    while start < text_length:
        end = min(start + chunk_size, text_length)
        
        # Look for natural break points (paragraphs, sentences, markdown headers)
        if end < text_length:
            # Search backward for break points
            boundary = max(
                text_with_placeholders.rfind('\n\n', start, end),
                text_with_placeholders.rfind('. ', start, end),
                text_with_placeholders.rfind('! ', start, end),
                text_with_placeholders.rfind('? ', start, end),
                text_with_placeholders.rfind('\n# ', start, end)  # Markdown headers
            )
            
            if boundary > start + int(chunk_size * 0.5):  # Only break if meaningful
                end = boundary + 1
                
        chunk = text_with_placeholders[start:end].strip()
        if chunk:
            text_chunks.append(chunk)
        
        start = end

    # Process chunks and link with code blocks
    text_docs = []
    code_docs = []
    code_ptr = 0  # Tracks position in code_blocks list
    
    for chunk in text_chunks:
        # Count code placeholders in this text chunk
        placeholder_count = chunk.count("[CODE_BLOCK]")
        
        # Get corresponding code blocks
        current_code_blocks = code_blocks[code_ptr:code_ptr + placeholder_count]
        code_ptr += placeholder_count
        
        # Create text document with cleaned content (remove placeholders)
        clean_text = chunk.replace("[CODE_BLOCK]", "").strip()
        if clean_text:
            text_docs.append(Document(
                page_content=clean_text,
                metadata={
                    "chunk_id": global_chunk_id,
                    "type": "text",
                    "num_code_blocks": len(current_code_blocks),
                    # "code_block_ids": list(range(code_ptr - placeholder_count, code_ptr)),
                    "url": url
                }
            ))
        
        # Create code documents for associated code blocks
        for code in current_code_blocks:
            code_docs.append(Document(
                page_content=code.strip(),
                metadata={
                    # "chunk_id": global_chunk_id,
                    "type": "code",
                    "parent_text_id": global_chunk_id,
                    "url": url
                }
            ))
        
        global_chunk_id += 1  # Increment globally to maintain uniqueness

    return text_docs, code_docs
