import matplotlib.pyplot as plt


def plot_hough(result):
    plt.figure()
    plt.imshow(
        result.hough,
        origin="lower",
        extent=[
            result.vx_values[0],
            result.vx_values[-1],
            result.vy_values[0],
            result.vy_values[-1],
        ],
        aspect="auto",
    )
    plt.xlabel(r"$v_x$")
    plt.ylabel(r"$v_y$")
    plt.title("Velocity Hough accumulator")
    plt.colorbar()
    plt.tight_layout()


def show_components(result):
    if result.components is None:
        raise ValueError("result does not contain reconstructed components")

    n = len(result.components)

    plt.figure(figsize=(4 * n, 4))

    for i, img in enumerate(result.components):
        plt.subplot(1, n, i + 1)
        plt.imshow(img, cmap="gray")
        plt.title(f"Component {i + 1}")
        plt.axis("off")

    plt.tight_layout()