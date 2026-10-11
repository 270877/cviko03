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
    Algoritmus k-means jako podtřída IterativeClustering.

    Tři krátké přepisy variačních bodů — iterační smyčka zůstává v základní třídě.
"""

from __future__ import annotations

import numpy as np

from src.base import IterativeClustering


class KMeans(IterativeClustering):
    """K-means shlukování s tvrdým přiřazením bodů k nejbližšímu těžišti.

    Dědí kompletní iterační smyčku z ``IterativeClustering``.
    Implementuje pouze:
    - přiřazení bodů jako ``argmin`` vzdáleností (tvrdé popisky),
    - přepočet těžišť jako prostý aritmetický průměr přiřazených bodů.
    """

    def _update_assignment(
        self, x: np.ndarray, centroids: np.ndarray
    ) -> np.ndarray:
        """Přiřadí každý bod k nejbližšímu těžišti (tvrdé přiřazení).

        Úkol:
            1. Vypočítejte matici vzdáleností bod-těžiště pomocí
               ``self._distances_to_centroids(x, centroids)``.
            2. Pro každý bod najděte index nejbližšího těžiště pomocí
               ``np.argmin`` podél osy těžišť (``axis=1``).

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        centroids:
            Aktuální těžiště tvaru ``(k, n_příznaků)``.

        Returns
        -------
        np.ndarray
            Tvrdé popisky shluků, tvar ``(n_bodů,)``, hodnoty 0 … k-1.
        """
        # assert: Ověřte, že x a centroids jsou 2D matice se stejným počtem příznaků
        assert isinstance(x, np.ndarray) and isinstance(centroids, np.ndarray)
        assert x.ndim == 2 and centroids.ndim == 2
        assert x.shape[1] == centroids.shape[1] #"Vstupy musí mít stejný počet příznaků."

        # 1. Vypočítejte matici vzdáleností
        distances = self._distances_to_centroids(x, centroids)
        # 2. Najděte index nejbližšího těžiště pro každý bod
        return np.argmin(distances, axis=1)


        raise NotImplementedError(
            "Úkol: implementujte KMeans._update_assignment() — použijte "
            "_distances_to_centroids a np.argmin pro tvrdé přiřazení."
        )

    def _update_centroids(
        self, x: np.ndarray, assignment: np.ndarray
    ) -> np.ndarray:
        """Přepočítá těžiště jako prostý průměr přiřazených bodů.

        Úkol:
            Pro každý shluk ``c`` v rozsahu 0 … k-1:
            1. Vyberte řádky ``x``, kde ``assignment == c``.
            2. Spočítejte průměr ``np.mean(..., axis=0)``.
            3. Pokud je shluk prázdný (žádný bod nepatří do ``c``), zachovejte
               staré těžiště ``self.centroids_[c]`` — zabraňuje NaN.

        Parameters
        ----------
        x:
            Příznakový matice tvaru ``(n_bodů, n_příznaků)``.
        assignment:
            Tvrdé popisky z ``_update_assignment``, tvar ``(n_bodů,)``.

        Returns
        -------
        np.ndarray
            Nová těžiště tvaru ``(k, n_příznaků)``.
        """
        # assert: Ověřte, že assignment je 1D pole a jeho délka odpovídá počtu bodů v x
        assert isinstance(x, np.ndarray) #"x musí být NumPy pole."
        assert isinstance(assignment, np.ndarray) #"assignment musí být NumPy pole."
        assert x.ndim == 2 #"x musí být 2D matice."
        assert assignment.ndim == 1 #"assignment musí být 1D pole."
        assert x.shape[0] == assignment.shape[0] #"Délka assignment musí odpovídat počtu bodů v x."

        n_features = x.shape[1]
        new_centroids = np.zeros((self.k, n_features), dtype=float)

        for c in range(self.k):
            # 1. Vyberte řádky x, kde assignment == c
            cluster_mask = (assignment == c)
            if np.any(cluster_mask):
                # 2. Spočítejte průměr bodů patřících do shluku
                new_centroids[c] = np.mean(x[cluster_mask], axis=0)
            else:
                # 3. Pokud je shluk prázdný, zachovejte staré těžiště
                if self.centroids_ is not None:
                    new_centroids[c] = self.centroids_[c]
                else:
                    # Fallback pro případ, že by se to stalo hned v prvním kroku
                    new_centroids[c] = np.zeros(n_features)

        return new_centroids

        raise NotImplementedError(
            "Úkol: implementujte KMeans._update_centroids() — průměr bodů "
            "přiřazených ke každému shluku. Ošetřete prázdné shluky."
        )

    def predict(self) -> np.ndarray:
        """Vrátí tvrdé popisky shluků uložené po volání ``fit``.

        Úkol:
            Vraťte ``self.assignment_`` — pole tvaru ``(n_bodů,)``
            s hodnotami 0 … k-1 přiřazenými v poslední iteraci ``fit``.

        Returns
        -------
        np.ndarray
            Tvrdé popisky shluků, tvar ``(n_bodů,)``.

        Raises
        ------
        RuntimeError
            Pokud ``fit`` nebyl dosud volán.
        """
        # assert: Ověřte, že fit() byl zavolán (self.assignment_ není None)
        if self.assignment_ is None:
            raise RuntimeError("Metoda fit() nebyla dosud volána.")
        return self.assignment_

        raise NotImplementedError(
            "Úkol: implementujte KMeans.predict() — vraťte self.assignment_ "
            "(tvrdé popisky uložené metodou fit)."
        )
