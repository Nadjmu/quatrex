def smallest_eigenvalue(A_csr, tol=1e-8, maxiter=100, gmres_tol=1e-10):
    n = A_csr.shape[0]

    # Force a supported floating/complex dtype.
    dtype = np.complex128 if np.iscomplexobj(A_csr.data) else np.float64
    A_csr = A_csr.astype(dtype, copy=False)
    A_csc = A_csr.tocsc()

    ilu = spilu(A_csc, drop_tol=1e-5, fill_factor=10)

    M = LinearOperator(
        A_csc.shape,
        matvec=ilu.solve,
        dtype=A_csc.dtype,
    )

    rng = np.random.default_rng(1234)

    if np.issubdtype(A_csr.dtype, np.complexfloating):
        x = rng.standard_normal(n) + 1j * rng.standard_normal(n)
        x = x.astype(A_csr.dtype)
    else:
        x = rng.standard_normal(n).astype(A_csr.dtype)

    x /= np.linalg.norm(x)

    for iteration in range(maxiter):
        y, info = gmres(
            A_csr,
            x,
            M=M,
            rtol=gmres_tol,
            atol=0.0,
        )

        if info != 0:
            raise RuntimeError(
                f"GMRES failed at inverse iteration {iteration}; info={info}"
            )

        y_norm = np.linalg.norm(y)
        if y_norm == 0 or not np.isfinite(y_norm):
            raise RuntimeError("Invalid vector returned by GMRES")

        y /= y_norm

        Ay = A_csr @ y

        # np.vdot conjugates its first argument.
        lam = np.vdot(y, Ay) / np.vdot(y, y)

        residual = Ay - lam * y
        relative_residual = np.linalg.norm(residual) / max(
            np.linalg.norm(Ay) + abs(lam) * np.linalg.norm(y),
            np.finfo(float).tiny,
        )

        print(
            f"iteration={iteration + 1}, "
            f"lambda={lam}, residual={relative_residual:.3e}"
        )

        if relative_residual < tol:
            return lam, y

        x = y

    raise RuntimeError("Inverse iteration did not converge")