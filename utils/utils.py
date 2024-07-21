import os


def log_to_csv(exp_name, filename, rel_errors, errors):
    """
    Log the relative errors and errors (MSE, Orth_A, Orth_B) of A and B to a CSV file
    """
    if not os.path.isfile(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                "Experiment,Rel_Error_A,Rel_Error_B,MSE_loss,Orth_loss_A,Orth_loss_B\n"
            )

    with open(filename, "a", encoding="utf-8") as f:
        f.write(
            f"{exp_name},{rel_errors['A'][-1]},{rel_errors['B'][-1]},{errors['MSE'][-1]},{errors['orth_A'][-1]},{errors['orth_B'][-1]}\n"
        )

    print("Relative Error of A: ", rel_errors["A"][-1])
    print("Relative Error of B: ", rel_errors["B"][-1])
    print("MSE Loss: ", errors["MSE"][-1])
    print(
        "Orthogonal Constraint Loss (A, B): ",
        errors["orth_A"][-1],
        errors["orth_B"][-1],
    )
