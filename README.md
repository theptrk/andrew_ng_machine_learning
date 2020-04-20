# andrew_ng_machine_learning

Read these Chapter 1-5 notes: [these notes for more info](https://theptrk.com/2020/02/12/notes-for-coursera-ml-course-week-1-5/)

Q: How to fix this mac error?
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

[1;2;3]~=[1;2;3]

% ans =
% 
%    0
%    0
%    0
```

### Create a results vector and iterate to collect info
```Octave
vector(i) = value
```
```Octave
results=zeros(4,1)

% iterate over a vector
for i=1:length(results)
    results(i) = i * 2 
end

% results =
% 
%     2
%     4
%     6
%     8
```

### Get the minimum value and index of a vector
```Octave
[min_value, min_index] = min([10;20;30])

% min_value =  10
% min_index =  1
```

### Get the max value and index of a matrix
This uses `find` as a shortcut
```
minVal = min(min(A));
[row_index, col_index] = find(A == minVal);
```

### Iterate over a matrix
```Octave
% Iterate over rows 
X_ = [1 2 3; 4 5 6]

for i = 1:size(X_, 1)
    row = X_(i, :)
    % do something with the row
end

```

### Filter a matrix by a boolean vector
```Octave
XX = [
    [1 2],
    [3 4],
    [5 6],
]
bv = [
    0
    1
    1
]
XX(bv, :)
ans =

   5.6586   4.8000
```