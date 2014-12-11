#include <stdio.h>
#include <math.h>
#include <gsl/gsl_linalg.h>
#include <gsl/gsl_matrix.h>
#include <gsl/gsl_vector.h>
#include <gsl/gsl_blas.h>
#include <libstandard.h>

gsl_matrix *gsl_inverse(gsl_matrix *m){
	int s=0;
	gsl_permutation * p = gsl_permutation_alloc (m->size1);
	gsl_matrix *inv=gsl_matrix_clone(m);
	gsl_matrix *r=gsl_matrix_alloc(inv->size1,inv->size2);
	gsl_linalg_LU_decomp(inv,p,&s);
	gsl_linalg_LU_invert(inv,p,r);
	gsl_matrix_free(inv);
	gsl_permutation_free(p);
	return r;
}
double m_log_normal_distribution(gsl_vector *data,gsl_vector *u,gsl_matrix *sigma){
	int i,j;
	gsl_matrix *mdata=gsl_matrix_alloc(1,data->size);
	gsl_matrix_set_row(mdata,1,data);
	gsl_matrix *tmp=gsl_matrix_alloc(sigma->size1,u->size);
	gsl_matrix *tmp2=gsl_matrix_alloc(sigma->size1,u->size);
	gsl_matrix *tmp3=gsl_matrix_alloc(sigma->size1,u->size);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,1.0, sigma, mdata,0.0, tmp);
	gsl_matrix_mul_constant(tmp,sqrt(2*M_PI));
	for(i=0;i<data->size;i++){
		gsl_vector_set(data,i,1/2*pow(log(gsl_vector_get(data,i))-gsl_vector_get(u,i),2));
	}
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,1.0,mdata, gsl_inverse(sigma), 0.0, tmp2);
	for(i=0;i<tmp->size1;i++){
		for(j=0;j<tmp->size1;j++){
			gsl_matrix_set(tmp2,i,j,exp(-gsl_matrix_get(tmp2,i,j)));
		}
	}
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,1.0,tmp, tmp2, 0.0, tmp3);
	return gsl_matrix_get(tmp3,0,0);
	
}
double log_normal_distribution(double x,double u,double sigma){
	return 1.0/(sqrt(2*M_PI)*sigma*x)*exp(-pow(log(x)-u,2)/2*sigma*sigma);
}
int main(void){
	double x,y;
	gsl_vector *data=gsl_vector_alloc(2);
	gsl_vector *u=gsl_vector_alloc(2);
	gsl_vector_set(u,0,0);
	gsl_vector_set(u,1,0);
	gsl_matrix *sigma=gsl_matrix_alloc(2,2);
	gsl_matrix_set(sigma,0,0,1);
	gsl_matrix_set(sigma,0,1,0);
	gsl_matrix_set(sigma,1,0,0);
	gsl_matrix_set(sigma,1,1,1);
	for(x=0.01;x<3.0;x+=0.01){
		for(y=0.01;y<3.0;y+=0.01){
			gsl_vector_set(data,0,x);
			gsl_vector_set(data,1,x);
			printf("%lf %lf %lf\n",x,y,m_log_normal_distribution(data,u,sigma));
		}
	}
	return 1;
}
