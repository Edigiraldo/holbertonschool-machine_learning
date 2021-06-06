#/usr/bin/env python3
"""Script for question answering."""

inp = ""

exit = ["exit", "quit", "goodbye", "bye"]
while True:
    inp = input("Q: ").lower()
    if inp in exit:
        print("A: Goodbye")
        break
    print("A:")
