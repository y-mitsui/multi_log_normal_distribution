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
double gsl_det(gsl_matrix *m){
	gsl_permutation * p = gsl_permutation_alloc (m->size1);
	gsl_matrix *lu=gsl_matrix_clone(m);
	int s=0;
	gsl_linalg_LU_decomp (lu,p,&s);           // LU分解
	double n = gsl_linalg_LU_det (lu,s);    // 行列式
	gsl_matrix_free(lu);
	gsl_permutation_free(p);
	
	return n;
}
double m_log_normal_distribution(gsl_vector *data,gsl_vector *u,gsl_matrix *sigma){
	int i;
	
	/* 1/det(sigma) */
	double sigmaDetInv=1.0/gsl_det(sigma);
	
	/* 1/(y_1*y_2...y_p) */
	double dataMulInv=1.0;
	for(i=0;i<data->size;i++){
		dataMulInv*=gsl_vector_get(data,i);
	}

	double constant=pow(1.0/sqrt(2*M_PI),data->size);
	
	/* log(y)-u */
	gsl_matrix *logDataSubUWidth=gsl_matrix_alloc(1,data->size);
	for(i=0;i<data->size;i++){
		gsl_matrix_set(logDataSubUWidth,0,i,log(gsl_vector_get(data,i))-gsl_vector_get(u,i));
	}
	/* (log(y)-u)*sigma^-1*/
	gsl_matrix *sigmaInv=gsl_inverse(sigma);
	gsl_matrix *dataSubUSigma=gsl_matrix_alloc(1,data->size);
	gsl_blas_dgemm (CblasNoTrans, CblasNoTrans,1.0,logDataSubUWidth, sigmaInv, 0.0, dataSubUSigma);

	/* mul log(y)-u*/
	gsl_matrix *solv1=gsl_matrix_alloc(1,1);
	gsl_blas_dgemm (CblasNoTrans, CblasTrans,1.0,dataSubUSigma, logDataSubUWidth, 0.0, solv1);
	
	double argExp=-1.0/2.0*gsl_matrix_get(solv1,0,0);

	return constant*sigmaDetInv*dataMulInv*exp(argExp);
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
	gsl_matrix_set(sigma,0,0,1.0);
	gsl_matrix_set(sigma,0,1,0.5);
	gsl_matrix_set(sigma,1,0,0.5);
	gsl_matrix_set(sigma,1,1,1.0);
	for(x=0.01;x<15.0;x+=0.2){
		for(y=0.01;y<15.0;y+=0.2){
			gsl_vector_set(data,0,x);
			gsl_vector_set(data,1,y);
			printf("%lf %lf %lf\n",x,y,m_log_normal_distribution(data,u,sigma));
		}
	}
	return 1;
}
