# from forge.sdk.abilities.registry import ability
from sympy import sympify


# @ability(
    name="evaluate_math_expression",
    description="Use sympy to evaluate a mathematical expression. Useful when you need to solve a computation.",
    parameters=[
        {
            "name": "expression",
            "description": "A string with an expression that can be evaluated by sympy.",
            "type": "string",
            "required": True,
        },
    ],
    output_type="str",
)
async def evaluate_math_expression(
    agent,
    task_id: str,
    expression: str,
) -> str:
    """
    Evaluate a mathematical expression using sympy.

    Args:
        expression (str): The expression to evaluate.

    Returns:
        str: The answer from sympy.
    """
    try:
        result = sympify(expression).evalf()  # Converts expression to a sympy expression and evaluates it
        return result
    except Exception as e:
        return str(e)
