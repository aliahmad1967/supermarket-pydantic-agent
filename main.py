import csv
import os
from pathlib import Path
from typing import Any

from dotenv import load_dotenv
from pydantic_ai import Agent, RunContext

PROJECT_ROOT = Path(__file__).resolve().parent
DATASET_PATH = PROJECT_ROOT / "supermarket_sales.csv"

load_dotenv(PROJECT_ROOT / ".env")
os.environ.setdefault("OLLAMA_BASE_URL", "http://localhost:11434/v1")
os.environ.setdefault("OLLAMA_API_KEY", "ollama")


def load_sales_data(path: Path = DATASET_PATH) -> list[dict[str, str]]:
    """Load the supermarket transactions from the local CSV file."""
    with path.open(newline="", encoding="utf-8-sig") as csv_file:
        return list(csv.DictReader(csv_file))


sales_data = load_sales_data()

agent = Agent(
    os.getenv("OLLAMA_MODEL", "ollama:gemma4:e4b"),
    deps_type=list[dict[str, str]],
    system_prompt=(
        "You are a supermarket sales analyst. Answer questions using only the local "
        "supermarket_sales.csv data. Always use the available tools for facts and "
        "calculations. State the relevant time period, filters, and units. If the "
        "data does not contain an answer, say so clearly."
    ),
)


@agent.tool
def dataset_overview(ctx: RunContext[list[dict[str, str]]]) -> dict[str, Any]:
    """Return the row count, columns, date range, and distinct dimension values."""
    rows = ctx.deps
    dates = sorted(row["Date"] for row in rows)
    return {
        "rows": len(rows),
        "columns": list(rows[0]) if rows else [],
        "date_range": [dates[0], dates[-1]] if dates else [],
        "branches": sorted({row["Branch"] for row in rows}),
        "cities": sorted({row["City"] for row in rows}),
        "product_lines": sorted({row["Product line"] for row in rows}),
        "payments": sorted({row["Payment"] for row in rows}),
    }


@agent.tool
def summarize_sales(
    ctx: RunContext[list[dict[str, str]]],
    group_by: str = "overall",
    city: str | None = None,
    product_line: str | None = None,
    payment: str | None = None,
) -> dict[str, Any]:
    """Calculate sales totals, averages, and counts, optionally grouped by a dimension."""
    allowed_groups = {
        "overall",
        "Branch",
        "City",
        "Customer type",
        "Gender",
        "Product line",
        "Payment",
    }
    group_names = {name.lower(): name for name in allowed_groups}
    group_by = group_names.get(group_by.lower(), group_by)
    if group_by not in allowed_groups:
        raise ValueError(
            f"group_by must be one of: {', '.join(sorted(allowed_groups))}"
        )

    rows = [
        row
        for row in ctx.deps
        if (city is None or row["City"].lower() == city.lower())
        and (
            product_line is None or row["Product line"].lower() == product_line.lower()
        )
        and (payment is None or row["Payment"].lower() == payment.lower())
    ]
    groups: dict[str, list[dict[str, str]]] = {"overall": rows}
    if group_by != "overall":
        groups = {}
        for row in rows:
            groups.setdefault(row[group_by], []).append(row)

    summaries = []
    for name, group in sorted(groups.items()):
        total = sum(float(row["Total"]) for row in group)
        summaries.append(
            {
                "group": name,
                "transactions": len(group),
                "total_sales": round(total, 2),
                "average_transaction": round(total / len(group), 2) if group else 0,
                "units_sold": sum(int(row["Quantity"]) for row in group),
                "gross_income": round(
                    sum(float(row["gross income"]) for row in group), 2
                ),
                "average_rating": round(
                    sum(float(row["Rating"]) for row in group) / len(group), 2
                )
                if group
                else 0,
            }
        )
    return {
        "filters": {"city": city, "product_line": product_line, "payment": payment},
        "summaries": summaries,
    }


@agent.tool
def find_transactions(
    ctx: RunContext[list[dict[str, str]]],
    city: str | None = None,
    product_line: str | None = None,
    payment: str | None = None,
    limit: int = 10,
) -> list[dict[str, str]]:
    """Find matching transactions and return a small, readable result set."""
    if not 1 <= limit <= 50:
        raise ValueError("limit must be between 1 and 50")
    matches = [
        row
        for row in ctx.deps
        if (city is None or row["City"].lower() == city.lower())
        and (
            product_line is None or row["Product line"].lower() == product_line.lower()
        )
        and (payment is None or row["Payment"].lower() == payment.lower())
    ]
    return [
        {
            key: row[key]
            for key in (
                "Invoice ID",
                "Branch",
                "City",
                "Product line",
                "Total",
                "Date",
                "Payment",
                "Rating",
            )
        }
        for row in matches[:limit]
    ]


def ask(question: str) -> str:
    """Ask the local agent a single question about the sales data."""
    return str(agent.run_sync(question, deps=sales_data).output)


def main() -> None:
    print(f"Loaded {len(sales_data):,} transactions from {DATASET_PATH.name}.")
    print("Ask a question about the sales data. Type 'exit' to quit.")
    while True:
        try:
            question = input("\nYou: ").strip()
        except EOFError, KeyboardInterrupt:
            print()
            break
        if question.lower() in {"exit", "quit"}:
            break
        if question:
            try:
                print(f"\nAgent: {ask(question)}")
            except Exception as error:
                print(f"\nAgent error: {error}")


if __name__ == "__main__":
    main()
