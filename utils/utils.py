import os


def log_to_csv(filename, exp_name, errors, args):
    """
    Log the args, relative errors and errors (MSE, Orth_A, Orth_B) of A and B to a CSV file
    """
    if not os.path.isfile(filename):
        with open(filename, "w", encoding="utf-8") as f:
            f.write(
                "Experiment,LearningRate_A,LearningRate_B,Orth_steps_per_iter,Rel_Error_W,MSE_loss,Orth_loss_A,Orth_loss_B\n"
            )

    rel_error = errors["rel_error_W"][-1] if "rel_error_W" in errors else "NAN"
    mse = errors["mse"][-1] if "mse" in errors else "NAN"
    tau = args.tau if "orth_A" in errors else "NAN"
    orth_A = errors["orth_A"][-1] if "orth_A" in errors else "NAN"
    orth_B = errors["orth_B"][-1] if "orth_B" in errors else "NAN"

    print("Experiment:", exp_name)
    print("Learning Rate A:", args.eta_A)
    print("Learning Rate B:", args.eta_B)
    print("Orthogonalization Steps per Iteration:", tau)
    print("Relative Error of (W, W*):", rel_error)
    print("MSE Loss:", mse)
    print("Orthogonal Constraint Loss A:", orth_A)
    print("Orthogonal Constraint Loss B:", orth_B)

    with open(filename, "a", encoding="utf-8") as f:
        f.write(
            f"{exp_name},{args.eta_A},{args.eta_B},{tau},{rel_error},{mse},{orth_A},{orth_B}\n"
        )
