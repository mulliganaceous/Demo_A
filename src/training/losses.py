"""Loss functions for PINN training.

Combines PDE residual loss, boundary condition loss, and initial condition loss
with configurable weights.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional

from ..physics.burgers import burgers_residual, burgers_initial_condition, burgers_boundary_condition
from ..physics.heat import heat_residual, heat_initial_condition, heat_boundary_condition


class PINNLoss(nn.Module):
    """Combined PINN loss: L = w_pde * L_pde + w_bc * L_bc + w_ic * L_ic.

    Args:
        pde_type: Type of PDE ('burgers' or 'heat').
        weights: Dict with loss weights {'pde': float, 'boundary': float, 'initial': float}.
        pde_params: Dict with PDE-specific parameters (e.g., viscosity, alpha).
    """

    def __init__(
        self,
        pde_type: str = "burgers",
        weights: Optional[Dict[str, float]] = None,
        pde_params: Optional[Dict[str, float]] = None,
    ):
        super().__init__()

        self.pde_type = pde_type
        self.weights = weights or {"pde": 1.0, "boundary": 10.0, "initial": 10.0}
        self.pde_params = pde_params or {}

    def compute_pde_loss(
        self,
        model: nn.Module,
        x_coll: torch.Tensor,
        t_coll: torch.Tensor,
    ) -> torch.Tensor:
        """Compute PDE residual loss (MSE of residual).

        Args:
            model: Neural network model.
            x_coll: Collocation x-coordinates.
            t_coll: Collocation t-coordinates.

        Returns:
            Scalar PDE loss.
        """
        if self.pde_type == "burgers":
            nu = self.pde_params.get("viscosity", 0.01 / 3.141592653589793)
            residual = burgers_residual(model, x_coll, t_coll, nu=nu)
        elif self.pde_type == "heat":
            alpha = self.pde_params.get("alpha", 0.01)
            residual = heat_residual(model, x_coll, t_coll, alpha=alpha)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

        return torch.mean(residual**2)

    def compute_boundary_loss(
        self,
        model: nn.Module,
        boundary_data: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        """Compute boundary condition loss.

        Args:
            model: Neural network model.
            boundary_data: Dict with boundary point coordinates.

        Returns:
            Scalar boundary loss.
        """
        x_left = boundary_data["x_left"]
        t_left = boundary_data["t_left"]
        x_right = boundary_data["x_right"]
        t_right = boundary_data["t_right"]

        # Predict at boundaries
        u_left = model(torch.cat([x_left, t_left], dim=1))
        u_right = model(torch.cat([x_right, t_right], dim=1))

        # Get target boundary values
        if self.pde_type == "burgers":
            bc_left, bc_right = burgers_boundary_condition(t_left)
        elif self.pde_type == "heat":
            bc_left, bc_right = heat_boundary_condition(t_left)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

        loss_left = torch.mean((u_left - bc_left) ** 2)
        loss_right = torch.mean((u_right - bc_right) ** 2)

        return loss_left + loss_right

    def compute_initial_loss(
        self,
        model: nn.Module,
        x_init: torch.Tensor,
        t_init: torch.Tensor,
    ) -> torch.Tensor:
        """Compute initial condition loss.

        Args:
            model: Neural network model.
            x_init: Initial condition x-coordinates.
            t_init: Initial condition t-coordinates (all zeros).

        Returns:
            Scalar initial condition loss.
        """
        u_pred = model(torch.cat([x_init, t_init], dim=1))

        if self.pde_type == "burgers":
            u_target = burgers_initial_condition(x_init)
        elif self.pde_type == "heat":
            u_target = heat_initial_condition(x_init)
        else:
            raise ValueError(f"Unknown PDE type: {self.pde_type}")

        return torch.mean((u_pred - u_target) ** 2)

    def forward(
        self,
        model: nn.Module,
        x_coll: torch.Tensor,
        t_coll: torch.Tensor,
        boundary_data: Dict[str, torch.Tensor],
        x_init: torch.Tensor,
        t_init: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """Compute total weighted loss.

        Returns:
            Dict with 'total', 'pde', 'boundary', 'initial' losses.
        """
        loss_pde = self.compute_pde_loss(model, x_coll, t_coll)
        loss_bc = self.compute_boundary_loss(model, boundary_data)
        loss_ic = self.compute_initial_loss(model, x_init, t_init)

        total = (
            self.weights["pde"] * loss_pde
            + self.weights["boundary"] * loss_bc
            + self.weights["initial"] * loss_ic
        )

        return {
            "total": total,
            "pde": loss_pde,
            "boundary": loss_bc,
            "initial": loss_ic,
        }
