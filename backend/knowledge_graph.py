import networkx as nx
from pathlib import Path


# ============================================================
# PROJECT PATH
# ============================================================

project_folder = Path(__file__).resolve().parent.parent

vectorstore_folder = project_folder / "vectorstore"

vectorstore_folder.mkdir(exist_ok=True)


# ============================================================
# CREATE KNOWLEDGE GRAPH
# ============================================================

def create_knowledge_graph():

    graph = nx.DiGraph()


    # ========================================================
    # COMPANY LEAVE POLICY
    # ========================================================

    graph.add_node(
        "Company Leave Policy",
        type="policy"
    )

    graph.add_node(
        "Annual Leave",
        type="benefit"
    )

    graph.add_node(
        "24 days",
        type="value"
    )

    graph.add_edge(
        "Company Leave Policy",
        "Annual Leave",
        relation="provides"
    )

    graph.add_edge(
        "Annual Leave",
        "24 days",
        relation="amount"
    )


    graph.add_node(
        "Work From Home",
        type="policy_rule"
    )

    graph.add_node(
        "2 days per week",
        type="value"
    )

    graph.add_node(
        "Manager Approval",
        type="requirement"
    )

    graph.add_edge(
        "Company Leave Policy",
        "Work From Home",
        relation="includes"
    )

    graph.add_edge(
        "Work From Home",
        "2 days per week",
        relation="limit"
    )

    graph.add_edge(
        "Work From Home",
        "Manager Approval",
        relation="requires"
    )


    graph.add_node(
        "Leave Request",
        type="process"
    )

    graph.add_node(
        "2 days advance",
        type="requirement"
    )

    graph.add_edge(
        "Company Leave Policy",
        "Leave Request",
        relation="includes"
    )

    graph.add_edge(
        "Leave Request",
        "2 days advance",
        relation="requires"
    )


    graph.add_node(
        "Emergency Leave",
        type="process"
    )

    graph.add_node(
        "Inform Manager",
        type="action"
    )

    graph.add_edge(
        "Company Leave Policy",
        "Emergency Leave",
        relation="includes"
    )

    graph.add_edge(
        "Emergency Leave",
        "Inform Manager",
        relation="requires"
    )


    # ========================================================
    # EMPLOYEE BENEFITS POLICY
    # ========================================================

    graph.add_node(
        "Employee Benefits Policy",
        type="policy"
    )

    graph.add_node(
        "Health Insurance",
        type="benefit"
    )

    graph.add_node(
        "Annual Performance Bonus",
        type="benefit"
    )

    graph.add_node(
        "Learning and Development Programs",
        type="benefit"
    )

    graph.add_edge(
        "Employee Benefits Policy",
        "Health Insurance",
        relation="provides"
    )

    graph.add_edge(
        "Employee Benefits Policy",
        "Annual Performance Bonus",
        relation="provides"
    )

    graph.add_edge(
        "Employee Benefits Policy",
        "Learning and Development Programs",
        relation="provides"
    )


    # ========================================================
    # HR POLICY
    # ========================================================

    graph.add_node(
        "HR Policy",
        type="policy"
    )

    graph.add_node(
        "Performance Reviews",
        type="process"
    )

    graph.add_node(
        "Twice a Year",
        type="frequency"
    )

    graph.add_edge(
        "HR Policy",
        "Performance Reviews",
        relation="includes"
    )

    graph.add_edge(
        "Performance Reviews",
        "Twice a Year",
        relation="frequency"
    )


    # ========================================================
    # IT SECURITY POLICY
    # ========================================================

    graph.add_node(
        "IT Security Policy",
        type="policy"
    )

    graph.add_node(
        "Security Incident",
        type="security_event"
    )

    graph.add_node(
        "Report to IT Immediately",
        type="action"
    )

    graph.add_edge(
        "IT Security Policy",
        "Security Incident",
        relation="covers"
    )

    graph.add_edge(
        "Security Incident",
        "Report to IT Immediately",
        relation="requires"
    )


    # ========================================================
    # SAVE GRAPH
    # ========================================================

    graph_file = vectorstore_folder / "company_knowledge_graph.graphml"

    nx.write_graphml(
        graph,
        graph_file
    )


    # ========================================================
    # DISPLAY INFORMATION
    # ========================================================

    print()
    print("=" * 60)
    print("COMPANY KNOWLEDGE GRAPH")
    print("=" * 60)

    print()

    print(
        "Total Nodes:",
        graph.number_of_nodes()
    )

    print(
        "Total Relationships:",
        graph.number_of_edges()
    )

    print()

    print("Relationships:")

    for source, target, data in graph.edges(data=True):

        print(
            f"{source} --[{data['relation']}]--> {target}"
        )

    print()

    print(
        "Knowledge graph saved to:"
    )

    print(
        graph_file
    )

    print("=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    create_knowledge_graph()