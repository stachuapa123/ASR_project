def lev_weighted(s1, s2, 
                          cost_delete=0.5,      # taniej usuwać
                          cost_insert=1.0,
                          cost_substitute=1.0):
    n, m = len(s1), len(s2)
    dp = [[0.0] * (m + 1) for _ in range(n + 1)]
    
    for i in range(n + 1):
        dp[i][0] = i * cost_delete
    for j in range(m + 1):
        dp[0][j] = j * cost_insert
    
    for i in range(1, n + 1):
        for j in range(1, m + 1):
            if s1[i-1] == s2[j-1]:
                dp[i][j] = dp[i-1][j-1]
            else:
                dp[i][j] = min(
                    dp[i-1][j] + cost_delete,
                    dp[i][j-1] + cost_insert,
                    dp[i-1][j-1] + cost_substitute,
                )
    
    return dp[n][m]