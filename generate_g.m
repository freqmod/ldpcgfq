#https://stackoverflow.com/questions/43887951/how-to-determine-ldpc-generator-matrix-form-parity-check-matrix-802-16e
pkg load communications



Q = 8;

#fileID =fopen('GF64_1134_20.alist');
#outFileName ='GF64_1134_20.mtx';

fileID =fopen('LDPC_N128_K64_GF256_UNBPB_exp.alist');
outFileName='LDPC_N128_K64_GF256_UNBPB_exp.mtx';

N = fscanf(fileID,'%d',1);
M = fscanf(fileID,'%d',1);
GF = fscanf(fileID,'%d',1);
dvmax = fscanf(fileID,'%d',1);
dcmax = fscanf(fileID,'%d',1);

dc=zeros(1,M);
dv=zeros(1,N);

matrix=zeros(M,N);
#matrix=matrix-1;
matrix2=matrix;

for i=1:N
   dv(i) = fscanf(fileID,'%d',1); 
end

for i=1:M
   dc(i) = fscanf(fileID,'%d',1); 
end

for i=1:N
    for j=1:dv(i)
       pos = fscanf(fileID,'%d',1); 
       value = fscanf(fileID,'%d',1);
       matrix(pos,i)=value;
    end
end


for i=1:M
    for j=1:dc(i)
       pos = fscanf(fileID,'%d',1); 
       value = fscanf(fileID,'%d',1);
       matrix2(i,pos)=value;
    end
end

noerror = isequal(matrix,matrix2);


fclose(fileID);

function F = make_gen_min(H)
    m = size(H, 1);
    n = size(H, 2);
    k = n - m;
    A = H(1:m, 1:k);
    B = H(1:m, k+1:n);
    F = (transpose(A) * inv(transpose(B)));
endfunction

function G = make_gen(H, Q)
    m = size(H, 1);
    n = size(H, 2);
    k = n - m;
    F = make_gen_min(H);
    G = [gf(eye(k), Q), F];

endfunction

#matrix2

#H = transpose(matrix2);
H = matrix2;
#max(H(:))
H = gf(H, Q);

G = make_gen(H, Q);
F = transpose(G(1:size(G,1), size(G,1)+1: size(G,2)));
#F
#G * transpose(H)

if(any(G * transpose(H)))
    disp ("Error: G * transpose(H) != 0");
else
    disp ("Note: G * transpose(H) == 0");
endif

outFileID = fopen(outFileName, 'w');

fprintf(outFileID, '%d %d %d\n',size(G,1), size(G,2), 2**Q);
for i=1:size(F,1)
    for j=1:size(F,2)
      fprintf(outFileID, '%d ', F(i,j).x);
    end
    fprintf(outFileID, '\n');
end
fclose(outFileID);
