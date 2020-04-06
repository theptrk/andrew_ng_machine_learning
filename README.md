# andrew_ng_machine_learning

Read: [these notes for more info](https://theptrk.com/2020/02/12/notes-for-coursera-ml-course-week-1-5/)

Q: How to fix mac error?
```
gnuplot> set terminal aqua enhanced title "Figure 1" ...
```
A: set `GNUTERM` to `X11`
```
> setenv GNUTERM qt
```

## Octave

### Get the error of a vector as a boolean
```Octave 
predictions ~= yval
```
```Octave
[1;2;3]~=[8;8;3]

% ans =
% 
%    1
%    1
%    0
```

### Create a results vector to collect info
```Octave
vector(i) = value
```
```Octave
results=zeros(8,1)

for i=1:8
    results(i) = i * 2
end

% results =
% 
%     2
%     4
%     6
%     8
%    10
```

### Get the minimum value and index of a vector
```Octave
[min_value, min_index] = min([10;20;30])

% min_value =  10
% min_index =  1
```

### Get the minimum value and index of a matrix
Here we:
- 1. get the column with minimum, derive the row index
- 2. get the row with minimum, derive the column index
```Octave
A=[10 20 30; 3 2 1; 11 22 33]

% A =
% 
%    10   20   30
%     3    2    1
%    11   22   33

%%% 1. Get the column with minimum, derive the row index  %%%

% 1a. This gets the colum with minimum

min(A,[], 2)
% ans =
% 
%    10
%     1
%    11

% 1b. This gets the colum with minimum AND derives the row index

[min_value, row_index] = min(min(A,[],2))
% min_value =  1
% row_index =  2

%%% 2. Get the row with minimum, derive the col index %%%

% 2a. This gets the row with minimum

min(A,[], 1)
% ans =
% 
%    3   2   1

% 2b. This gets the row with minimum AND derives the col index

[min_value, col_index] = min(min(A,[], 1))
% min_value =  1
% col_index =  3
```

### Get the max value and index of a matrix
This uses `find` as a shortcut
```
minVal = min(min(A));
[row_index, col_index] = find(A == minVal);
```