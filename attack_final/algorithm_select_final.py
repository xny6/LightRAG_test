# import json
# from collections import defaultdict

# def load_data(json_path):
#     with open(json_path, 'r', encoding='utf-8') as f:
#         return json.load(f)

# def get_relation_id(relation):
#     """
#     把一个关系唯一表示为三元组字符串。
#     这样即便相同语义的关系重复出现在多个问题中也能识别出来。
#     """
#     return f"{relation['Original Entity']} --> {relation['Anchor Entity']}: {relation['Relationship']}"

# def select_target_relations(data, max_relations=None):
#     # 建立 Map: relation_id -> set of question indices
#     relation_to_questions = defaultdict(set)
    
#     for q_idx, item in enumerate(data):
#         for relation in item.get("Relation", []):
#             rid = get_relation_id(relation)
#             relation_to_questions[rid].add(q_idx)

#     # 初始化
#     selected_relations = set()
#     covered_questions = set()
#     total_questions = set(range(len(data)))

#     # 贪心选择
#     while covered_questions != total_questions:
#         # 选出覆盖未覆盖问题最多的关系
#         best_relation = None
#         best_coverage = set()
        
#         for rid, questions in relation_to_questions.items():
#             new_coverage = questions - covered_questions
#             if len(new_coverage) > len(best_coverage):
#                 best_relation = rid
#                 best_coverage = new_coverage
        
#         if not best_relation:
#             break  # 无法继续覆盖更多问题

#         selected_relations.add(best_relation)
#         covered_questions.update(best_coverage)

#         if max_relations and len(selected_relations) >= max_relations:
#             break

#     return selected_relations

# # 示例使用
# if __name__ == "__main__":
#     json_path = "/home/NingyuanXiao/LightRAG_test/attack_final/filtered_relationships_structured.json"  # 替换为你的实际路径
#     data = load_data(json_path)
#     selected = select_target_relations(data, max_relations=10)

#     print("Selected Target Relations (Top-k):")
#     for i, r in enumerate(selected):
#         print(f"{i+1}. {r}")




import json
from collections import defaultdict

def load_data(json_path):
    """读取 JSON 数据"""
    with open(json_path, 'r', encoding='utf-8') as f:
        return json.load(f)

def get_relation_id(relation):
    """
    修改后的关系标识方式：只用 'Relationship' 字段作为唯一 ID
    """
    return relation['Relationship'].strip()

def select_target_relations(data, max_relations=None):
    """
    实现论文 Algorithm 1：贪心选择最能覆盖问题的关系集合
    """
    # 建立 Map: relationship_text -> set of question indices
    relation_to_questions = defaultdict(set)
    
    for q_idx, item in enumerate(data):
        for relation in item.get("Relation", []):
            rid = get_relation_id(relation)
            relation_to_questions[rid].add(q_idx)

    # 初始化
    selected_relations = set()
    covered_questions = set()
    total_questions = set(range(len(data)))

    # 贪心选择关系
    while covered_questions != total_questions:
        best_relation = None
        best_coverage = set()
        
        for rid, questions in relation_to_questions.items():
            new_coverage = questions - covered_questions
            if len(new_coverage) > len(best_coverage):
                best_relation = rid
                best_coverage = new_coverage
        
        if not best_relation:
            break  # 没有更多关系能覆盖新问题了

        selected_relations.add(best_relation)
        covered_questions.update(best_coverage)

        if max_relations and len(selected_relations) >= max_relations:
            break

    return selected_relations

# 示例调用
if __name__ == "__main__":
    json_path = "/home/NingyuanXiao/LightRAG_test/attack_final/filtered_relationships_structured.json"  # 替换成你的实际文件路径
    data = load_data(json_path)
    selected = select_target_relations(data, max_relations=30)

    print("Selected Target Relations (based on 'Relationship'):")
    for i, r in enumerate(selected):
        print(f"{i+1}. {r}")
