<h2> Supply code of the publication:</h2>

B. Cagri Sarar, Ivan Giorgio, Gino D’Ovidio, Francesco Fabbrocino, B. Emek Abali, and M. Erden Yildizdag (2027). Voxel-based strain-gradient modeling of TPMS metamaterials for advanced railway passenger vehicle applications

#This code underlies the GNU General Public License, http://www.gnu.org/licenses/gpl-3.0.en.html

For parameter determination, use

docker run --rm -ti ghcr.io/scientificcomputing/fenics:2025-10-09

for strain gradient elasticity calculation, use

docker run --rm -ti firedrakeproject/firedrake:latest 

in docker, it is possible to mount a local folder with

-v /mylocalfolder/:/shared/

then in the docker 

cd /shared
ls

will show the local folder in the computer and results will be written there (update /home/... in the code).
