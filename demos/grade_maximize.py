from dp import DPArray, display

def grade_maximize(n, H, grades):
    A = DPArray(shape=(n + 1, H+1))
    
    for h in range(H+1):
        # No classes left, the rest of the hours do nothing.
        A[0, h] = 0
    
    for i in range(1, n + 1):
        # Restricting only to classes 1 through i.
        for h in range(H+1):
            # Restricting only to study up to h hours.
            
            # Solution 1 (most likely to be implemented)
            t = grades[i-1][0] + A[i-1, h]
            for h_prime in range(1, h+1):
                # Find the optimal amount study time for the ith class.
                t = max(t, grades[i-1][h_prime] + A[i-1, h - h_prime])
            A[i, h] = t
            
            # Solution 2 (using dpvis version of max)
            # indices = [(i-1, h_prime) for h_prime in range(h+1)]
            # elements = [grades[i-1][h_prime] + A[i-1, h - h_prime] for (_, h_prime) in indices]
            # A[i, h] = A.max(indices=indices, elements=elements)
    
    display(A)

    return (1 / n) * A[n, H]

NUM_CLASSES = 3
NUM_HOURS = 6
grades = [[1, 1.7, 2.7, 3.3, 4, 3.3, 4], 
          [2.3, 3.3, 3, 3.3, 2.7, 3.3, 3.7], 
          [3, 3.3, 3.3, 3.7, 3.3, 2.7, 4]]

print(grade_maximize(NUM_CLASSES, NUM_HOURS, grades))

