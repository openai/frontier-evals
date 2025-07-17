import argparse
import json
import threading
import unicodedata
from dataclasses import replace
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request
from paperbench.judge.graded_task_node import GradedTaskNode, update_all_grades
from paperbench.rubric.tasks import VALID_TASK_CATEGORIES, TaskNode, generate_task_category
from paperbench.rubric.utils import get_openai_client, random_id

app = Flask(__name__, static_folder="static")
lock = threading.Lock()


def validate_openai_key() -> bool:
    try:
        get_openai_client()
        return True

    except Exception as e:
        return False


def sanitize_text(text: Any) -> Any:
    """
    Sanitize text input by normalizing Unicode characters,
    removing non-breaking spaces, and trimming whitespace,
    """
    if not isinstance(text, str):
        return text  # If it's not a string, return as-is

    normalized_text = unicodedata.normalize("NFKC", text)

    # weird chars not covered by NFKC
    sanitized_text = normalized_text.replace("\u200b", " ")  # non-breaking spaces
    sanitized_text = sanitized_text.replace("\u2019", "'")  # curly single quotes

    return sanitized_text.strip()


def sanitize_request_data(data: Any) -> Any:
    """Recursively sanitize all text fields in a JSON object."""
    if isinstance(data, dict):
        return {k: sanitize_request_data(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [sanitize_request_data(v) for v in data]
    elif isinstance(data, str):
        return sanitize_text(data)
    return data


@app.route("/")
def index():
    with lock:  # Use lock to support concurrent requests
        try:
            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                data = json.load(f)

            task_node = app.config["NODE_TYPE"].from_dict(data)
            if app.config.get("GRADED", False):
                task_node = update_all_grades(task_node)
                with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                    json.dump(task_node.to_dict(), f, indent=4)

            template = "rubric.html"

            return render_template(
                template,
                task_node=task_node,
                valid_task_categories=VALID_TASK_CATEGORIES,
                use_api=app.config["USE_API"],
            )
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Error fetching the task tree: {e}",
                }
            )


@app.route("/delete_node", methods=["GET"])
def delete_node():
    """Deletes the node with the given `node_id`."""

    with lock:  # Use lock to support concurrent requests
        node_id = sanitize_text(request.args.get("node_id"))

        try:
            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            old_root = app.config["NODE_TYPE"].from_dict(rubric)
            old_parent = old_root.get_parent(node_id)

            if old_root.id == node_id:
                return jsonify(
                    {
                        "status": "error",
                        "message": "Cannot delete the root node.",
                    }
                )

            # Handle the edge case where the parent of `node_id` is now a leaf node, after
            # having deleted `node_id`. We have to generate a new `task_category` since this is
            # required for all leaf nodes.
            if len(old_parent.sub_tasks) == 1:
                new_task_category = None

                if app.config["USE_API"]:
                    new_task_category = generate_task_category(old_parent)

                new_parent = replace(old_parent, task_category=new_task_category, sub_tasks=[])
                new_root = old_root.replace(old_parent.id, new_parent)
            else:
                new_root = old_root.delete(node_id)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Error deleting node {node_id}: {e}",
                }
            )


@app.route("/update_requirements", methods=["POST"])
def update_requirements():
    """Updates the requirements for the given `node_id`."""
    with lock:  # Use lock to support concurrent requests
        data: dict = sanitize_request_data(request.json)
        node_id = data.get("node_id")
        new_requirements = data.get("requirements")

        try:
            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            old_root = app.config["NODE_TYPE"].from_dict(rubric)
            old_node = old_root.find(node_id)
            new_node = old_node.set_requirements(new_requirements)
            new_root = old_root.replace(node_id, new_node)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Error updating requirements for node {node_id}: {str(e)}",
                }
            )


@app.route("/add_sub_task", methods=["POST"])
def add_sub_task():
    """Adds a new sub-task to the given parent node."""

    with lock:  # Use lock to support concurrent requests
        data = sanitize_request_data(request.json)
        parent_id = data.get("parent_id")
        new_requirements = data.get("requirements")

        try:
            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            root = app.config["NODE_TYPE"].from_dict(rubric)
            parent_node = root.find(parent_id)

            if args.graded:
                new_sub_task = app.config["NODE_TYPE"](
                    id=random_id(),
                    task_category=None,
                    requirements=new_requirements,
                    weight=1,
                    score=0.0,
                    explanation="",
                    judge_metadata={},
                    valid_score=True,
                    sub_tasks=[],
                )
            else:
                new_sub_task = app.config["NODE_TYPE"](
                    id=random_id(),
                    task_category=None,
                    requirements=new_requirements,
                    weight=1,
                    sub_tasks=[],
                )

            if app.config["USE_API"]:
                new_task_category = generate_task_category(new_sub_task)
                new_sub_task = new_sub_task.set_task_category(new_task_category)

            new_parent_node = parent_node.add_sub_task(new_sub_task)
            new_root = root.replace(parent_id, new_parent_node)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Error adding sub-task to node {parent_id}: {str(e)}",
                }
            )


@app.route("/update_weight", methods=["POST"])
def update_weight():
    """Updates the weight for the given `node_id`."""
    with lock:  # Use lock to support concurrent requests
        data = sanitize_request_data(request.json)
        node_id = data.get("node_id")
        new_weight = data.get("weight")

        try:
            new_weight = int(new_weight)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            old_root = app.config["NODE_TYPE"].from_dict(rubric)
            old_node = old_root.find(node_id)
            new_node = old_node.set_weight(new_weight)
            new_root = old_root.replace(node_id, new_node)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Error updating weight for node {node_id}: {str(e)}",
                }
            )


@app.route("/update_task_category", methods=["POST"])
def update_task_category():
    """Updates the task category for the given node_id."""
    with lock:
        data = sanitize_request_data(request.json)
        node_id = data.get("node_id")
        new_task_category = data.get("category")

        try:
            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            old_root = app.config["NODE_TYPE"].from_dict(rubric)
            old_node = old_root.find(node_id)
            new_node = old_node.set_task_category(new_task_category)
            new_root = old_root.replace(node_id, new_node)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Error updating task category for node {node_id}: {str(e)}",
                }
            )


@app.route("/move_node", methods=["POST"])
def move_node():
    """Moves a node up or down relative to its siblings."""
    with lock:
        data = sanitize_request_data(request.json)
        node_id = data.get("node_id")
        direction = data.get("direction")  # "up" or "down"

        try:
            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            root = app.config["NODE_TYPE"].from_dict(rubric)

            try:
                parent = root.get_parent(node_id)
            except ValueError as e:
                return jsonify({"status": "error", "message": "Cannot move the root node."})

            current_idx = next(i for i, task in enumerate(parent.sub_tasks) if task.id == node_id)

            if direction == "up" and current_idx > 0:
                new_idx = current_idx - 1
            elif direction == "down" and current_idx < len(parent.sub_tasks) - 1:
                new_idx = current_idx + 1
            else:
                return jsonify(
                    {"status": "error", "message": "Cannot move node further in that direction."}
                )

            # Swap nodes
            new_sub_tasks = list(parent.sub_tasks)
            new_sub_tasks[current_idx], new_sub_tasks[new_idx] = (
                new_sub_tasks[new_idx],
                new_sub_tasks[current_idx],
            )

            new_parent = parent.set_sub_tasks(new_sub_tasks)
            new_root = root.replace(parent.id, new_parent)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})

        except Exception as e:
            return jsonify({"status": "error", "message": f"Error moving node {node_id}: {str(e)}"})


@app.route("/duplicate_node", methods=["POST"])
def duplicate_node():
    """Duplicates the node with the given `node_id` and adds it as a sibling."""
    with lock:
        data = sanitize_request_data(request.json)
        node_id = data.get("node_id")

        try:
            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            old_root = app.config["NODE_TYPE"].from_dict(rubric)
            if old_root.id == node_id:
                return jsonify({"status": "error", "message": "Cannot duplicate the root node."})

            node_to_duplicate = old_root.find(node_id)
            parent = old_root.get_parent(node_id)
            duplicate_node = node_to_duplicate.duplicate_with_new_ids()

            new_parent = parent.add_sub_task(duplicate_node)
            new_root = old_root.replace(parent.id, new_parent)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify(
                {"status": "error", "message": f"Error duplicating node {node_id}: {str(e)}"}
            )


@app.route("/move_node_to_parent", methods=["POST"])
def move_node_to_parent():
    """Moves a node to become a child of another node."""
    with lock:
        data = sanitize_request_data(request.json)
        node_id = data.get("node_id")
        new_parent_id = data.get("new_parent_id")

        try:
            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            root = app.config["NODE_TYPE"].from_dict(rubric)
            node_to_move = root.find(node_id)

            if not root.contains(node_id):
                return jsonify(
                    {"status": "error", "message": f"Node with ID '{node_id}' not found."}
                )
            if not root.contains(new_parent_id):
                return jsonify(
                    {
                        "status": "error",
                        "message": f"Parent node with ID '{new_parent_id}' not found.",
                    }
                )
            if root.id == node_id:
                return jsonify({"status": "error", "message": "Cannot move the root node."})
            if node_to_move.contains(new_parent_id):
                return jsonify(
                    {
                        "status": "error",
                        "message": "Cannot move a node to itself or to one of its descendants.",
                    }
                )

            new_root = root.delete(node_id)
            new_parent = new_root.find(new_parent_id)
            new_new_parent = new_parent.add_sub_task(node_to_move)
            new_root = new_root.replace(new_parent_id, new_new_parent)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify({"status": "error", "message": f"Error moving node {node_id}: {str(e)}"})


@app.route("/update_score", methods=["POST"])
def update_score():
    with lock:
        try:
            data = sanitize_request_data(request.get_json())
            node_id = data["node_id"]
            new_score = float(data["score"])
            if not (0 <= new_score <= 1):
                return jsonify({"status": "error", "message": "Score must be between 0 and 1"})

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            old_root = app.config["NODE_TYPE"].from_dict(rubric)
            old_node = old_root.find(node_id)
            new_node = old_node.set_score(new_score)
            new_root = old_root.replace(node_id, new_node)

            new_root = update_all_grades(new_root)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})

        except Exception as e:
            return jsonify({"status": "error", "message": str(e)})


@app.route("/update_explanation", methods=["POST"])
def update_explanation():
    """Updates the explanation for the given node_id."""
    with lock:
        data = sanitize_request_data(request.json)
        node_id = data.get("node_id")
        new_explanation = data.get("explanation")

        try:
            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "r") as f:
                rubric = json.load(f)

            old_root = app.config["NODE_TYPE"].from_dict(rubric)
            old_node = old_root.find(node_id)
            new_node = old_node.set_explanation(new_explanation)
            new_root = old_root.replace(node_id, new_node)

            with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
                json.dump(new_root.to_dict(), f, indent=4)

            return jsonify({"status": "success"})
        except Exception as e:
            return jsonify(
                {
                    "status": "error",
                    "message": f"Error updating explanation for node {node_id}: {str(e)}",
                }
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Rubric Viewer")
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument(
        "--path-to-paper",
        type=Path,
        help="Path to the paper directory, which should include `paper.pdf`, `paper.md` and `config.yaml`.",
    )
    parser.add_argument(
        "--rubric-file-name",
        type=str,
        help="Name of the rubric file. Defaults to `rubric.json`.",
        default="rubric.json",
    )
    parser.add_argument(
        "--graded",
        action="store_true",
        help="Use flag if rubric contains GradedTaskNodes.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="o1",
        help="Model to use for the LLM",
    )
    parser.add_argument(
        "--use-api",
        action="store_true",
        help="Use flag if you want to enable OAI API use e.g. for expanding nodes",
    )
    args = parser.parse_args()

    app.config["USE_API"] = False
    if args.use_api:
        openai_api_key_is_set = validate_openai_key()
        if not openai_api_key_is_set:
            raise ValueError(
                "Cannot use `--use-api` without setting the OPENAI_API_KEY environment variable."
            )
        app.config["USE_API"] = True

    app.config["MODEL"] = args.model
    app.config["GRADED"] = args.graded
    app.config["NODE_TYPE"] = GradedTaskNode if args.graded else TaskNode
    app.config["PATH_TO_PAPER"] = args.path_to_paper
    app.config["RUBRIC_FILE_NAME"] = args.rubric_file_name
    app.config["PAPER_MD_FILE_NAME"] = "paper.md"

    if not (app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"]).exists():
        if args.graded:
            empty_task = app.config["NODE_TYPE"](
                id=random_id(),
                task_category=None,
                requirements="The core contributions of the paper have been reproduced.",
                weight=1,
                score=0.0,
                explanation="",
                judge_metadata={},
                valid_score=True,
                sub_tasks=[],
            )
        else:
            empty_task = app.config["NODE_TYPE"](
                id=random_id(),
                task_category=None,
                requirements="The core contributions of the paper have been reproduced.",
                weight=1,
                sub_tasks=[],
            )
        with open(app.config["PATH_TO_PAPER"] / app.config["RUBRIC_FILE_NAME"], "w") as f:
            json.dump(empty_task.to_dict(), f, indent=4)

    app.run(debug=True)
