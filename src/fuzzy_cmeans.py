# -*- coding: utf-8 -*-

"""
Created on 25. 08. 2026 at 20:59:30

Author: Richard Redina
Email: 195715@vut.cz
Affiliation:
         International Clinical Research Center, Brno
         Brno University of Technology, Brno
GitHub: RicRedi

(._.)
 <|>
_/|_

Description:
    Algoritmus fuzzy c-means jako podtřída IterativeClustering.

    Klíčový rozdíl oproti k-means: bod patří do všech shluků zároveň s různou
    měrou členství. Parametr ``q`` řídí míru „rozmazání" hranic shluků.
"""

from __future__ import annotations

import numpy as np

from src import (
    IterativeClustering,
    Distance,
    Initializer,
    )


class FuzzyCMeans(IterativeClustering):
    """Fuzzy c-means shlukování s měkkým přiřazením bodů.

    Každý bod má přiřazen vektor členství délky ``k``, jehož složky udávají
    míru příslušnosti do každého shluku. Součet členství každého bodu je 1.

    Parametr ``q`` (fuzzifikace) ovlivňuje míru překryvu shluků:
    - ``q → 1``:  přiblíží se k-means (tvrdé hranice),
    - ``q = 2``:  standardní fuzzy c-means,
    - ``q > 2``:  velmi měkké hranice (shluky se silně překrývají).
    """

    def __init__(
        self,
        k: int,
        distance: Distance,
        initializer: Initializer,
        q: float = 2.0,
        max_iter: int = 100,
    ) -> None:
        """Inicializuje FCM s parametrem fuzifikace ``q``.

        Parameters
        ----------
        k:
            Počet shluků.
        distance:
            Instance metriky vzdálenosti (injektovaná závislost).
        initializer:
            Instance inicializační strategie (injektovaná závislost).
        q:
            Parametr fuzifikace — musí být > 1. Výchozí hodnota 2.0
            odpovídá standardnímu fuzzy c-means.
        max_iter:
            Maximální počet iterací.
        """
        # assert: Ověřte, že parametr fuzifikace q je větší než 1
        assert isinstance(q, (int, float)) and q > 1.0 #"Parametr fuzifikace q musí být větší než 1."
        super().__init__(k, distance, initializer, max_iter)
        self.q: float = q

    def _update_assignment(
        self, x: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Vypočítá matici členství pomocí standardního vzorce FCM.

        Úkol:
            Implementujte výpočet matice členství ``U`` tvaru ``(n_bodů, k)``:

            Pro každý bod ``i`` a shluk ``c``:

            .. math::
                U_{ic} = \\frac{1}{\\sum_{j=1}^{k}
                \\left(\\frac{d_{ic}}{d_{ij}}\\right)^{\\frac{2}{q-1}}}

            kde :math:`d_{ic}` je vzdálenost bodu ``i`` od těžiště ``c``.

            Každý řádek ``U[i]`` musí sumovat na 1.

            Pozor na dělení nulou:
            Pokud bod leží přesně na těžišti (``d_{ic} = 0``), přiřaďte
            tomuto bodu plné členství v daném shluku (``U[i, c] = 1``,
            ostatní složky = 0) a přeskočte standardní výpočet.

            Nápověda: exponent je ``2 / (self.q - 1)``.

        Parameters
        ----------
        X:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        centroids:
            Aktuální těžiště tvaru ``(k, n_příznaků)``.

        Returns
        -------
        np.ndarray
            Matice členství tvaru ``(n_bodů, k)``, každý řádek sumuje na 1.
        """
        # assert: Ověřte, že x a centroids jsou 2D matice se stejným počtem příznaků
        assert isinstance(x, np.ndarray) and isinstance(centroids, np.ndarray)
        assert x.ndim == 2 and centroids.ndim == 2
        assert x.shape[1] == centroids.shape[1] #"Vstupy musí mít stejný počet příznaků."

        n_samples = x.shape[0]
        distances = self._distances_to_centroids(x, centroids)
        u = np.zeros((n_samples, self.k), dtype=float)
        exponent = 2.0 / (self.q - 1.0)

        for i in range(n_samples):
            # Kontrola, zda některá vzdálenost je 0 (bod leží přesně na těžišti)
            zero_mask = (distances[i] == 0)
            if np.any(zero_mask):
                # Bod leží na těžišti (případně na více těžištích) -> plné členství
                u[i, zero_mask] = 1.0 / np.sum(zero_mask)
            else:
                # Standardní FCM vzorec pre výpočet matice členstva
                # Pro každý shluk c: sum_j (d_ic / d_ij)^(2 / (q - 1))
                for c in range(self.k):
                    ratios = distances[i, c] / distances[i]
                    denominator = np.sum(ratios ** exponent)
                    u[i, c] = 1.0 / denominator
        return u

        raise NotImplementedError(
            "Úkol: implementujte FuzzyCMeans._update_assignment() — matici "
            "členství FCM. Viz docstring pro vzorec a ošetření dělení nulou."
        )

    def _update_centroids(
        self, x: np.ndarray, assignment: np.ndarray
    ) -> np.ndarray:
        """Přepočítá těžiště jako vážený průměr bodů s vahami z matice členství.

        Úkol:
            Pro každý shluk ``c`` spočítejte vážený průměr:

            .. math::
                \\text{centroid}_c =
                \\frac{\\sum_{i} U_{ic}^q \\cdot X_i}{\\sum_{i} U_{ic}^q}

            kde ``U[:, c]`` je sloupec matice členství pro shluk ``c``
            a váhy jsou ``U[:, c] ** self.q``.

            Nápověda: váhy mají tvar ``(n_bodů,)`` — pro vážený průměr
            rozměrů ``(n_příznaků,)`` je třeba správné přetvarování.

        Parameters
        ----------
        X:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        assignment:
            Matice členství tvaru ``(n_bodů, k)`` — výstup ``_update_assignment``.

        Returns
        -------
        np.ndarray
            Nová těžiště tvaru ``(k, n_příznaků)``.
        """
        # assert: Ověřte, že assignment (matice členství U) je 2D pole tvaru (n_bodů, k)
        assert isinstance(x, np.ndarray) #"x musí být NumPy pole."
        assert isinstance(assignment, np.ndarray) #"assignment musí být NumPy pole."
        assert x.ndim == 2 #"x musí být 2D matice."
        assert assignment.ndim == 2 #"assignment (matice členství U) musí být 2D matice."
        assert assignment.shape[0] == x.shape[0] #"Počet řádků assignment musí odpovídat počtu bodů v x."
        assert assignment.shape[1] == self.k #"Počet sloupců assignment musí odpovídat k."

        n_features = x.shape[1]
        new_centroids = np.zeros((self.k, n_features), dtype=float)

        # Váhy umocněné na q: U_{ic}^q
        u_q = assignment ** self.q

        for c in range(self.k):
            weights = u_q[:, c]  # tvar (n_bodů,)
            weight_sum = np.sum(weights)

            if weight_sum > 0:
                # Vážený průměr: součet(weights[:, None] * x, axis=0) / součet(weights)
                new_centroids[c] = np.sum(weights[:, None] * x, axis=0) / weight_sum
            else:
                # Ošetření nulového součtu vah (zachování starého těžiště)
                if self.centroids_ is not None:
                    new_centroids[c] = self.centroids_[c]
                else:
                    new_centroids[c] = np.zeros(n_features)
        return new_centroids

        raise NotImplementedError(
            "Úkol: implementujte FuzzyCMeans._update_centroids() — vážený průměr "
            "bodů s vahami z matice členství umocněné na self.q."
        )

    def predict(self) -> np.ndarray:
        """Převede měkkou matici členství na tvrdé popisky shluků (argmax).

        Úkol:
            Vraťte ``np.argmax(self.assignment_, axis=1)`` — pro každý bod
            vyberte shluk s nejvyšším členstvím.

            Tím jsou výstupy FCM kompatibilní s ``silhouette.py``, které
            očekává tvrdé popisky stejně jako k-means.

        Returns
        -------
        np.ndarray
            Tvrdé popisky shluků, tvar ``(n_bodů,)``, hodnoty 0 … k-1.

        Raises
        ------
        RuntimeError
            Pokud ``fit`` nebyl dosud volán.
        """
        # assert: Ověřte, že fit() byl zavolán a self.assignment_ je 2D matice členství
        assert self.assignment_ is not None #"Metoda fit() nebyla dosud volána."
        assert isinstance(self.assignment_, np.ndarray) and self.assignment_.ndim == 2 #self.assignment_ musí být 2D matice členství."

        # Vrátí index maximálního členství pro každý bod podél osy shluků (axis=1)
        return np.argmax(self.assignment_, axis=1)

        raise NotImplementedError(
            "Úkol: implementujte FuzzyCMeans.predict() — vraťte argmax "
            "matice členství self.assignment_ podél osy shluků (axis=1)."
        )
