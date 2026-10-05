#!/bin/bash
# Stage 3b -- mirror-image detection for ONE protein's 50 quench replicas.
# Submitted by 10_sweep_mirror.sh.
#
#   sbatch -J mi_A_08 -o NSC_ENT/08_P32179/quench/mirror_analysis.out \
#          -e NSC_ENT/08_P32179/quench/mirror_analysis.err scripts/job_mirror.sh NSC_ENT/08_P32179
#
# ALWAYS pass -o/-e, for the same reason job_q.sh does: a #SBATCH directive
# cannot reference $1, so without them every run drops a pair of files in the
# sim root instead of beside the data it describes.
#
# ALL detection is topo's own `python -m topo.analysis.mirror` -- this file only
# fans that CLI out over the 50 split replicas and collects its verdicts. No
# metric is computed here. topo's defaults are used throughout, including
# --q-ss-only (Q restricted to secondary-structure contacts) and
# --summary-last-frames 133 (the trailing window the TRAJECTORY MIRROR verdict
# is taken over), which is passed explicitly so the log records it.
#
# STRIDE comes from setup/<UNIPROT>_stride.dat, cached at build time, so the
# SS assignment matches stage 1 and 2 and STRIDE need not exist on the node.
#
# CPU only: this reads finished trajectories and does no simulation, so it must
# not sit on a GPU while stage-2 jobs are queueing for one.
#SBATCH --partition=standard
#SBATCH --account=epo2_cr_default
#SBATCH -N 1
#SBATCH -n 1
#SBATCH --cpus-per-task=16
#SBATCH --mem=16G
#SBATCH -t 04:00:00
#SBATCH -o logs/mirror_%x.out
#SBATCH -e logs/mirror_%x.err

set -eo pipefail

# Same ordering as job_q.sh: `set -u` only AFTER conda activation, because the
# topo-prod activate.d hook dereferences an unset $PYTHONPATH.
source /storage/group/epo2/default/qzv5006/miniconda3/etc/profile.d/conda.sh
conda activate topo-prod
set -u

# Locate the sim root. Under sbatch the script runs from SLURM's spooled copy,
# so BASH_SOURCE is useless there and SLURM_SUBMIT_DIR is the anchor (as in
# job_q.sh); run by hand from an OOD/interactive session and SLURM_SUBMIT_DIR
# points at the dashboard instead, so neither candidate can be trusted blindly.
# Take the first one that actually looks like the sim root.
SIM=""
for cand in "${SLURM_SUBMIT_DIR:-}" "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." 2>/dev/null && pwd)"; do
    if [[ -n "$cand" && -f "${cand}/scripts/job_mirror.sh" && -d "${cand}/NSC_ENT" ]]; then
        SIM="$cand"; break
    fi
done
[[ -n "$SIM" ]] || { echo "cannot locate sim root" >&2; exit 1; }
cd "$SIM"

TAG="$1"                                   # e.g. NSC_ENT/08_P32179
LAST_FRAMES="${LAST_FRAMES:-133}"          # topo default
NPROC="${SLURM_CPUS_PER_TASK:-4}"

PDIR="${SIM}/${TAG}"
UNIPROT="$(basename "$PDIR")"; UNIPROT="${UNIPROT#*_}"
TRAJ="${PDIR}/quench/traj"
OUT="${TRAJ}/mirror"

echo "host    : $(hostname)"
echo "protein : ${TAG} (${UNIPROT})"
echo "topo    : $(python -c 'import topo,os;print(os.path.dirname(topo.__file__))')"
echo "window  : last ${LAST_FRAMES} frames (topo default)"
echo "started : $(date)"

mkdir -p "$OUT"

# One topo CLI call per replica, NPROC at a time. Each call rebuilds the
# reference from the same PDB + cached STRIDE, so replicas stay comparable.
run_one() {
    local i="$1"
    local dcd="${TRAJ}/traj_${i}.dcd"
    [[ -f "$dcd" ]] || { echo "missing $dcd" >&2; return 1; }
    python -m topo.analysis.mirror \
        -r "${PDIR}/setup/${UNIPROT}.pdb" \
        -f "$dcd" \
        -p "${TRAJ}/traj.psf" \
        -s "${PDIR}/setup/${UNIPROT}_stride.dat" \
        -o "${OUT}/mirror_${i}.csv" \
        --summary-last-frames "$LAST_FRAMES" \
        > "${OUT}/mirror_${i}.log" 2>&1
}
export -f run_one
export TRAJ PDIR UNIPROT OUT LAST_FRAMES

seq 0 49 | xargs -P "$NPROC" -I{} bash -c 'run_one {}'

# Collect topo's own per-trajectory verdict from each replica's log.
{
    printf 'replica\tQ_tail\tK_tail\tRMSD_ratio_tail\tmirror_frames_tail\tTRAJECTORY_MIRROR\n'
    for i in $(seq 0 49); do
        log="${OUT}/mirror_${i}.log"
        q=$(awk -F: '/mean Q \(tail\)/          {gsub(/ /,"",$2); print $2}' "$log")
        k=$(awk -F: '/mean K \(tail\)/          {gsub(/ /,"",$2); print $2}' "$log")
        r=$(awk -F: '/mean RMSD_ratio \(tail\)/ {gsub(/ /,"",$2); print $2}' "$log")
        m=$(awk -F: '/mirror frames \(tail\)/   {gsub(/ /,"",$2); print $2}' "$log")
        v=$(awk '/TRAJECTORY MIRROR/ {print $4}' "$log")
        printf '%s\t%s\t%s\t%s\t%s\t%s\n' "$i" "$q" "$k" "$r" "$m" "$v"
    done
} > "${OUT}/verdicts.tsv"

nmir=$(awk -F'\t' 'NR>1 && $6=="True"' "${OUT}/verdicts.tsv" | wc -l)
echo
echo "MIRROR REPLICAS: ${nmir}/50   (topo TRAJECTORY MIRROR, last ${LAST_FRAMES} frames)"
awk -F'\t' 'NR>1 && $6=="True" {printf "%s ", $1}' "${OUT}/verdicts.tsv"; echo
echo "wrote ${OUT}"
echo "finished: $(date)"
