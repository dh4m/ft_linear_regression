import sys
import json
from pathlib import Path

def estimate_price(mile: float):
    file = Path("train_parameter.json")
    if file.exists():
        with open("train_parameter.json", 'r') as f:
            parameter = json.load(f)
            theta0 = parameter['theta0']
            theta1 = parameter['theta1']
    else:
        theta0 = 0
        theta1 = 0
    return theta0 + theta1 * mile


def main():
    """
    main() -> None

    main function, Entrypoint when this module is executed directly.
    """
    assert len(sys.argv) == 2, "bad argument"
    try:
        mile = float(sys.argv[1])
    except:
        print("bad argument", file=sys.stderr)
        return
    print("estimated price for distance is: ", estimate_price(mile))


if __name__ == "__main__":
    try:
        main()
    except AssertionError as e:
        print(f"AssertionError: {e}", file=sys.stderr)
    except Exception as e:
        print(f"Exception Occurred!", file=sys.stderr)
        print(f"{type(e).__name__}: {e}", file=sys.stderr)
