import os
from typing import TypedDict, List, Dict, Any, Annotated
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langgraph.graph import StateGraph, END
import operator

# ========== 1. 定义 Agent 状态 ==========
class AgentState(TypedDict):
    # 输入
    documents: List[Dict[str, str]]   # 每个元素: {"title": "合同A", "content": "..."}
    # 中间结果
    extracted_entities: List[Dict]    # 信息提取Agent的输出
    compliance_issues: List[Dict]     # 合规检查Agent的输出
    cross_issues: List[Dict]          # 交叉验证Agent的输出
    # 最终报告
    final_report: str
    # 流程控制
    next_step: str

# ========== 2. 初始化 LLM ==========
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

# ========== 3. 定义 Agent 节点 ==========

# 3.1 信息提取 Agent
def extract_entities_node(state: AgentState) -> AgentState:
    docs = state["documents"]
    all_entities = []
    
    for doc in docs:
        prompt = f"""
        你是一个信息提取专家。从以下文档中提取关键实体，返回 JSON 数组，每个元素包含: 
        - type (如: 金额, 日期, 责任方, 义务条款, 权利条款)
        - value (具体内容)
        - source_doc (文档标题)
        - context (原文片段，不超过30字)
        
        文档标题: {doc['title']}
        文档内容: {doc['content'][:2000]}  # 防止超长
        
        只返回 JSON，不要有其他文字。
        """
        response = llm.invoke([HumanMessage(content=prompt)])
        # 简单解析，实际应使用 json.loads 并处理异常
        try:
            import json
            entities = json.loads(response.content)
            if isinstance(entities, list):
                all_entities.extend(entities)
        except:
            all_entities.append({"error": response.content})
    
    state["extracted_entities"] = all_entities
    state["next_step"] = "compliance_check"
    return state

# 3.2 合规检查 Agent（包含长链推理）
def compliance_check_node(state: AgentState) -> AgentState:
    entities = state["extracted_entities"]
    issues = []
    
    # 构造一个法规知识库（模拟）
    regulations = {
        "违约金": "根据《民法典》第585条，违约金超过实际损失30%可请求减少。",
        "单方变更": "格式条款中单方变更服务内容需提供合理通知期，否则可能无效。",
    }
    
    for ent in entities:
        if "条款" in ent.get("type", ""):
            clause = ent.get("value", "")
            # 长链推理：逐个条件判断
            reasoning_chain = []
            risk_level = "low"
            
            # 步骤1: 检测关键词
            if "单方调整" in clause or "任意变更" in clause:
                reasoning_chain.append("检测到单方变更条款")
                # 步骤2: 检查是否提供通知期
                if "通知" not in clause and "告知" not in clause:
                    reasoning_chain.append("未声明任何通知义务")
                    # 步骤3: 引用法规
                    reasoning_chain.append(f"法规参考: {regulations.get('单方变更', '')}")
                    risk_level = "high"
                else:
                    reasoning_chain.append("虽可单方变更但提供了通知期 → 中风险")
                    risk_level = "medium"
            
            if reasoning_chain:
                issues.append({
                    "doc": ent.get("source_doc"),
                    "clause": clause,
                    "risk": risk_level,
                    "reasoning": " → ".join(reasoning_chain),
                    "suggestion": "建议增加乙方异议权或限定变更范围" if risk_level=="high" else "确认通知期是否合理"
                })
    
    state["compliance_issues"] = issues
    state["next_step"] = "cross_validation"
    return state

# 3.3 交叉验证 Agent
def cross_validation_node(state: AgentState) -> AgentState:
    docs = state["documents"]
    entities = state["extracted_entities"]
    cross_issues = []
    
    # 按实体类型分组
    amounts = [e for e in entities if e.get("type") == "金额"]
    dates = [e for e in entities if e.get("type") == "日期"]
    
    # 检查付款金额一致性
    if len(amounts) > 1:
        values = [a.get("value") for a in amounts]
        if len(set(values)) > 1:
            cross_issues.append({
                "type": "金额冲突",
                "details": f"不同文档中标明的金额不一致: {values}",
                "suggestion": "请确认以哪个文档为准，通常在主合同中定义。"
            })
    
    # 检查日期逻辑
    for date_ent in dates:
        if "验收" in date_ent.get("context", "") and "付款" in date_ent.get("context", ""):
            # 简单模拟更复杂的逻辑推理
            cross_issues.append({
                "type": "日期依赖",
                "details": f"文档 '{date_ent['source_doc']}' 中的付款日期依赖于验收日期，需确保验收条款明确。",
                "suggestion": "检查验收流程是否有明确定义的时间限制。"
            })
    
    state["cross_issues"] = cross_issues
    state["next_step"] = "generate_report"
    return state

# 3.4 生成报告节点
def generate_report_node(state: AgentState) -> AgentState:
    report_lines = []
    report_lines.append("# 智能文档审阅报告\n")
    
    # 合规问题
    if state["compliance_issues"]:
        report_lines.append("## 合规风险\n")
        for issue in state["compliance_issues"]:
            report_lines.append(f"- **{issue['risk']}风险** - 文档: {issue['doc']}\n")
            report_lines.append(f"  条款: {issue['clause'][:100]}\n")
            report_lines.append(f"  推理链: {issue['reasoning']}\n")
            report_lines.append(f"  建议: {issue['suggestion']}\n\n")
    else:
        report_lines.append("## 合规风险\n未发现明显合规问题。\n")
    
    # 交叉验证问题
    if state["cross_issues"]:
        report_lines.append("## 跨文档一致性问题\n")
        for issue in state["cross_issues"]:
            report_lines.append(f"- {issue['type']}: {issue['details']}\n")
            report_lines.append(f"  建议: {issue['suggestion']}\n\n")
    
    # 实体概览
    report_lines.append("## 提取的关键实体（摘要）\n")
    for ent in state["extracted_entities"][:5]:
        report_lines.append(f"- {ent.get('type')}: {ent.get('value')} (来自 {ent.get('source_doc')})\n")
    
    state["final_report"] = "".join(report_lines)
    state["next_step"] = "end"
    return state

# ========== 4. 构建 LangGraph 工作流 ==========
builder = StateGraph(AgentState)

builder.add_node("extract_entities", extract_entities_node)
builder.add_node("compliance_check", compliance_check_node)
builder.add_node("cross_validation", cross_validation_node)
builder.add_node("generate_report", generate_report_node)

builder.set_entry_point("extract_entities")
builder.add_edge("extract_entities", "compliance_check")
builder.add_edge("compliance_check", "cross_validation")
builder.add_edge("cross_validation", "generate_report")
builder.add_edge("generate_report", END)

graph = builder.compile()

# ========== 5. 模拟运行 ==========
if __name__ == "__main__":
    # 模拟输入文档
    sample_docs = [
        {
            "title": "主合同",
            "content": "甲方应于验收合格后15日内支付乙方合同总金额100万元。甲方有权单方调整服务内容，无需提前通知。"
        },
        {
            "title": "附件B-技术规格",
            "content": "验收后30天内付款。服务内容变更需提前5个工作日书面通知乙方。"
        }
    ]
    
    initial_state = {
        "documents": sample_docs,
        "extracted_entities": [],
        "compliance_issues": [],
        "cross_issues": [],
        "final_report": "",
        "next_step": "start"
    }
    
    final_state = graph.invoke(initial_state)
    print(final_state["final_report"])
