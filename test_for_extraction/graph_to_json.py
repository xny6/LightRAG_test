import networkx as nx
import json

def graphml_to_json(graphml_path, json_path):
    # 读取 GraphML 文件
    G = nx.read_graphml(graphml_path)

    # 提取节点信息
    nodes = []
    for node_id, node_data in G.nodes(data=True):
        node_entry = {
            "id": node_id,
            "attributes": dict(node_data)
        }
        nodes.append(node_entry)

    # 提取边信息
    edges = []
    for source, target, edge_data in G.edges(data=True):
        edge_entry = {
            "source": source,
            "target": target,
            "attributes": dict(edge_data)
        }
        edges.append(edge_entry)

    # 合并为一个 JSON 字典
    graph_json = {
        "nodes": nodes,
        "edges": edges
    }

    # 写入 JSON 文件
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(graph_json, f, indent=2, ensure_ascii=False)

    print(f"Graph successfully converted to JSON and saved to {json_path}")


# 示例调用
if __name__ == "__main__":
    graphml_file = "/home/NingyuanXiao/LightRAG_test/working_dir_advanced_ollama/graph_chunk_entity_relation.graphml"  # 请替换为你的实际文件路径
    json_output_file = "/home/NingyuanXiao/LightRAG_test/test_for_extraction/graph_output.json"
    graphml_to_json(graphml_file, json_output_file)
