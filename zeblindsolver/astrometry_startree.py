from __future__ import annotations

import math
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from astropy.io import fits


def _native_array(hdu: fits.BinTableHDU, dtype: str, count: int) -> np.ndarray:
    name = hdu.columns.names[0]
    raw = hdu.data[name].tobytes()
    return np.frombuffer(raw, dtype=dtype, count=count).copy()


def _radec_to_xyz(ra_deg: float, dec_deg: float) -> np.ndarray:
    ra = math.radians(float(ra_deg))
    dec = math.radians(float(dec_deg))
    cosdec = math.cos(dec)
    return np.asarray(
        [cosdec * math.cos(ra), cosdec * math.sin(ra), math.sin(dec)],
        dtype=np.float64,
    )


def xyz_to_radec(xyz: np.ndarray) -> np.ndarray:
    points = np.asarray(xyz, dtype=np.float64)
    ra = np.degrees(np.arctan2(points[:, 1], points[:, 0])) % 360.0
    dec = np.degrees(
        np.arctan2(points[:, 2], np.hypot(points[:, 0], points[:, 1]))
    )
    return np.column_stack((ra, dec))


@dataclass(frozen=True)
class StartreeQuery:
    xyz: np.ndarray
    star_ids: np.ndarray
    sweep: np.ndarray


@dataclass
class AstrometryStartree:
    data: np.ndarray
    lr: np.ndarray
    split: np.ndarray
    sweep: np.ndarray
    minval: np.ndarray
    scale: float
    ninterior: int

    @classmethod
    def open(cls, path: str | Path) -> "AstrometryStartree":
        p = Path(path).expanduser().resolve()
        with fits.open(p, mode="readonly", memmap=True) as hdul:
            header_hdu = next(
                hdu
                for hdu in hdul
                if str(hdu.header.get("AN_FILE", "")).strip() == "SKDT"
                and str(hdu.header.get("KDT_NAME", "")).strip() == "stars"
            )
            header = header_hdu.header
            ndata = int(header["KDT_NDAT"])
            ndim = int(header["KDT_NDIM"])
            nnodes = int(header["KDT_NNOD"])
            if ndim != 3:
                raise ValueError(f"unsupported Astrometry startree dimension: {ndim}")
            if str(header.get("KDT_INT", "")).strip().lower() != "u32":
                raise ValueError("Astrometry startree requires u32 tree storage")
            if str(header.get("KDT_DATA", "")).strip().lower() != "u32":
                raise ValueError("Astrometry startree requires u32 point storage")

            by_column = {
                hdu.columns.names[0]: hdu
                for hdu in hdul
                if getattr(hdu, "columns", None) is not None
                and getattr(hdu.columns, "names", None)
            }
            ninterior = (nnodes - 1) // 2
            nbottom = (nnodes + 1) // 2
            lr = _native_array(by_column["kdtree_lr_stars"], "<u4", nbottom)
            split = _native_array(
                by_column["kdtree_split_stars"], "<u4", ninterior
            )
            ranges = _native_array(
                by_column["kdtree_range_stars"], "<f8", ndim * 2 + 1
            )
            data = _native_array(
                by_column["kdtree_data_stars"], "<u4", ndata * ndim
            ).reshape(ndata, ndim)
            sweep = np.asarray(
                by_column["sweep"].data["sweep"], dtype=np.uint8
            ).copy()
        if sweep.shape[0] != ndata:
            raise ValueError(
                f"invalid Astrometry sweep length: {sweep.shape[0]} != {ndata}"
            )
        return cls(
            data=data,
            lr=lr,
            split=split,
            sweep=sweep,
            minval=np.asarray(ranges[:ndim], dtype=np.float64),
            scale=float(ranges[-1]),
            ninterior=ninterior,
        )

    def query_radec(
        self,
        ra_deg: float,
        dec_deg: float,
        radius_deg: float,
    ) -> StartreeQuery:
        center = _radec_to_xyz(ra_deg, dec_deg)
        chord_radius = 2.0 * math.sin(0.5 * math.radians(float(radius_deg)))
        return self.query_xyz(center, chord_radius * chord_radius)

    def query_xyz(
        self,
        center_xyz: np.ndarray,
        radius2: float,
    ) -> StartreeQuery:
        center = np.asarray(center_xyz, dtype=np.float64).reshape(3)
        maxd2 = float(radius2)
        if not np.all(np.isfinite(center)) or not np.isfinite(maxd2) or maxd2 < 0:
            raise ValueError("invalid Astrometry startree query")

        invscale = 1.0 / self.scale
        converted = (center - self.minval) * self.scale
        use_integer_query = bool(
            np.all(converted >= 0.0)
            and np.all(converted <= float(np.iinfo(np.uint32).max))
        )
        tquery = converted.astype(np.uint32) if use_integer_query else None
        tlinf = int(math.ceil(math.sqrt(maxd2) * self.scale))
        dimmask = 3
        splitmask = 0xFFFFFFFC

        stack = [0]
        found: list[int] = []
        while stack:
            node = stack.pop()
            if node >= self.ninterior:
                leaf = node - self.ninterior
                left = 0 if leaf == 0 else int(self.lr[leaf - 1]) + 1
                right = int(self.lr[leaf])
                if right < left:
                    continue
                raw_points = self.data[left : right + 1]
                points = raw_points.astype(np.float64) * invscale + self.minval
                keep = np.sum((points - center[None, :]) ** 2, axis=1) <= maxd2
                if np.any(keep):
                    found.extend((np.flatnonzero(keep) + left).tolist())
                continue

            packed = int(self.split[node])
            dim = packed & dimmask
            split = packed & splitmask
            left_child = 2 * node + 1
            right_child = left_child + 1
            if use_integer_query and tquery is not None:
                query_value = int(tquery[dim])
                if query_value < split:
                    stack.append(left_child)
                    if split - query_value <= tlinf:
                        stack.append(right_child)
                else:
                    stack.append(right_child)
                    if query_value - split <= tlinf:
                        stack.append(left_child)
            else:
                split_external = float(split) * invscale + float(self.minval[dim])
                maxdist = math.sqrt(maxd2)
                if center[dim] < split_external:
                    stack.append(left_child)
                    if split_external - center[dim] <= maxdist:
                        stack.append(right_child)
                else:
                    stack.append(right_child)
                    if center[dim] - split_external <= maxdist:
                        stack.append(left_child)

        star_ids = np.asarray(found, dtype=np.int64)
        if star_ids.size == 0:
            return StartreeQuery(
                xyz=np.empty((0, 3), dtype=np.float64),
                star_ids=star_ids,
                sweep=np.empty((0,), dtype=np.uint8),
            )
        xyz = (
            self.data[star_ids].astype(np.float64) * invscale
            + self.minval[None, :]
        )
        return StartreeQuery(
            xyz=xyz,
            star_ids=star_ids,
            sweep=self.sweep[star_ids],
        )


@lru_cache(maxsize=4)
def load_astrometry_startree(path: str) -> AstrometryStartree:
    return AstrometryStartree.open(path)
