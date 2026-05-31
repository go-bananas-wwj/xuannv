#!/usr/bin/env python3
"""Modify patent document by replacing annotation-like text with formal content."""

import html
import re

def encode_for_xml(text):
    """Encode Chinese characters for Word XML."""
    result = []
    for char in text:
        if ord(char) > 127:
            result.append(f'&#{ord(char)};')
        else:
            result.append(char)
    return ''.join(result)

def main():
    xml_path = 'unpacked_doc/word/document.xml'
    
    with open(xml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Decode to work with Chinese text
    decoded = html.unescape(content)
    
    # === Modification 1: Replace annotation in 1.1 section ===
    # Original: "从技术的角度，之前的模型是什么样子，他们有什么问题。时空嵌入是什么，解释一下。"
    # Replace with formal technical description
    old_text1 = '从技术的角度，之前的模型是什么样子，他们有什么问题。时空嵌入是什么，解释一下。'
    new_text1 = ('时空嵌入（Spatio-Temporal Embedding）是指将多源异构遥感数据编码为紧凑、'
                 '低维且可复用的向量表征的技术。该向量同时承载空间信息（地表覆盖类型、空间结构）'
                 '和时间信息（季节变化、年际变化、突发事件），使模型能够理解地表状态的动态演变。'
                 '现有模型主要存在以下技术问题：')
    
    if old_text1 in decoded:
        decoded = decoded.replace(old_text1, new_text1)
        print(f'Replaced: {old_text1[:30]}...')
    else:
        print(f'WARNING: Could not find: {old_text1[:30]}...')
    
    # Original: "行业痛点不写了，写技术问题。"
    # This should be removed as it's a note to self
    old_text2 = '行业痛点不写了，写技术问题。'
    new_text2 = ''  # Remove entirely
    
    if old_text2 in decoded:
        decoded = decoded.replace(old_text2, new_text2)
        print(f'Removed: {old_text2}')
    else:
        print(f'WARNING: Could not find: {old_text2}')
    
    # === Modification 2: Replace/integrate annotation in 3.3 section ===
    # The text "输入头的设计保证了系统的可扩展性——新增数据源时，只需增加对应的专用输入头，无需修改下游模块"
    # is actually good content, but it reads like a note. Let's refine it.
    old_text3 = '输入头的设计保证了系统的可扩展性——新增数据源时，只需增加对应的专用输入头，无需修改下游模块。'
    new_text3 = '该架构设计使系统具备良好的数据源可扩展性：当需要引入新型数据源时，仅需增加对应的专用输入头模块，无需对下游的特征提取、嵌入生成及任务解码模块进行任何修改，从而降低了系统扩展的开发成本和维护复杂度。'
    
    if old_text3 in decoded:
        decoded = decoded.replace(old_text3, new_text3)
        print(f'Replaced: {old_text3[:30]}...')
    else:
        print(f'WARNING: Could not find: {old_text3[:30]}...')
    
    # Re-encode Chinese characters for XML
    # We need to encode all Chinese characters back to numeric entities
    encoded = ''
    for char in decoded:
        if ord(char) > 127:
            encoded += f'&#{ord(char)};'
        else:
            encoded += char
    
    # Write back
    with open(xml_path, 'w', encoding='utf-8') as f:
        f.write(encoded)
    
    print('\nModifications complete!')

if __name__ == '__main__':
    main()
