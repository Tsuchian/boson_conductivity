
#include "warehouse.h"
#include "mpi.h"
using namespace std;
int main(int argc, char *argv[])
{
    if (argc != 8)
    {
        cerr << "Error : Wrong number of argument assigned!" << endl
             << "Usage: " << argv[0] << " <Dimension of hamiltonian> <# of k> t1 t2 kT <mag> <density> " << endl
             << "Example: " << argv[0] << " 2 10000 30 20 10 1 0.075 1>opt.txt 2>opt.log" << endl;
        abort();
    }

    tool sol;
    init init_matrix;
    int num_k = atoi(argv[2]);
    int N = atoi(argv[1]);
    double t1 = atof(argv[3]);
    double t2 = atof(argv[4]);
    double mag = atof(argv[6]);
    double rho = atof(argv[7]);
    double kT = atof(argv[5]);
    double e_min = -5 * t1;
    double e_max = 10 * t1;
    mat2D_arma a(N, 1, mag, t1, t2, t1);
    double ifboson = -1;
    int id, np, ierr;
    double wtime;
    cerr.precision(16);
    ierr = MPI_Init(NULL, NULL);
    if (ierr != 0)
    {
        cerr << endl;
        cerr << "MPI - Fatal error!" << endl;
        exit(1);
    }
    ierr = MPI_Comm_size(MPI_COMM_WORLD, &np);
    ierr = MPI_Comm_rank(MPI_COMM_WORLD, &id);
    cerr << "id: " << id << endl;
    if (id == 0)
    {
        cerr << "# The number of process is: " << np << endl;
        cerr << "# num_k:" << num_k << endl;
        cerr << a;
    }
    if (id == 0)
    {
        wtime = MPI_Wtime();
    }
    //sol.dos3d_M(a, sol.dos,num_k,e_min,e_max,0.1);
    mat eigvals;
    //vector<vec>eigvals;
    sol.gene_eigs_oneband(eigvals, a, num_k);
    //cerr<<"# The calculation of dos is done!"<<endl;
    //double mu=sol.chempot(sol.dos,rho,kT,0.000001);cerr<<"mu:"<<mu<<endl;
    long double mu = sol.chempot_real_3d_gen_fermi(a, eigvals, rho, kT, 10e-10);
    cerr << "mu:" << mu << endl;
    long double bott = eigvals.min() - a.tz;
    //long double bott=sol.chempot(sol.dos,rho,0.01,0.000001);
    //long double bott=sol.chempot_real_3d_gen(a,eigvals,rho,0.1,0.000001);
    cerr << "# bott of the band:" << bott << endl;
    //cout<<"omega " <<"sigmaxx_inter "<<"sigmaxy_inter "<<"sigmaxx_intra "<<"simgxy_intra "<<endl;
    int n_p = 1200;
    int nr = 4;
    vector<double> sig_local(n_p * nr), sig_all(n_p * nr);
    //bool ifc=sol.if_condensed_quasi2D(a,eigvals,rho,kT,10e-10);
    //if (ifc &&a.tz != 0 && ifboson == 1)
    //{
    //    mu=bott;
    //}

    for (int i = id; i < n_p; i = i + np)
    {
        double omega = double(i) / 4;
        sig_local[i * nr] = omega;
        cx_vec temp(2 * N + 2, fill::zeros);
        sol.Optcond(N, a.mag, t1, t2, num_k, kT, mu, omega, temp, bott);
        double sigmaxx = 0;
        for (int j = 0; j < a.dim; j++)
        {
            sigmaxx += real(temp(j));
            //cerr<<temp[j]<<" ";
        }
        //cerr<<endl;
        //cout<<omega<<" "<<sigmaxx<<endl;
        sig_local[i * nr + 1] = sigmaxx;
        sig_local[i * nr + 2] = real(temp(2 * N));
        sig_local[i * nr + 3] = real(temp(2 * N + 1));
        //cout<<omega<<" "<<real(temp[0])<<" "<<real(temp[1])<<" "<<real(temp[2])<<" "<<real(temp[3])<<" "<<real(temp[0]+temp[2])<<endl;temp.clear();
    }
    MPI_Reduce(sig_local.data(), sig_all.data(), n_p * nr, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);
    if (id == 0)
    {
        double sum = 0;
        double sumintra = 0;
        double suminter = 0;
        for (int i = 0; i < n_p; i++)
        {
            //cout<<sig_all[i*n_p]<<" ";
            //sum += sig_all[i * nr + 1];
            for (int j = 0; j < nr; j++)
            {
                cout << sig_all[i * nr + j] << " ";
            }
            sum += sig_all[i * nr + 1];
            sumintra += sig_all[i * 4 + 2];
            suminter += sig_all[i * 4 + 3];
            cout << endl;
        }
        cerr << "# check the sum rule : " << sum << endl;
        cerr << "# check the sumintra rule : " << sumintra << endl;
        cerr << "# check the suminter rule : " << suminter << endl;
    }
    if (id == 0)
    {
        cerr.precision(14);
        wtime = MPI_Wtime() - wtime;
        cerr << "USED TIME: " << wtime << endl;
    }
    MPI_Finalize();
}
