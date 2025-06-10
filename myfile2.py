{
  "time_functions": {
    "TimeBetween": {
      "syntax": "TimeBetween(start_date, end_date, time_level, include_end)",
      "example": "TimeBetween(20120101,20171231,[Time].[Year], false)",
      "keywords": ["between", "range", "from", "to", "during", "date", "year", "month", "day", "period"],
      "description": "Filter data within specific date ranges",
      "use_case": "Use when query mentions date ranges, time periods, or 'between' dates"
    },
    "TRENDNUMBER": {
      "syntax": "TRENDNUMBER(measure, time_level, periods, trend_type)",
      "example": "TRENDNUMBER([Measures.PROFIT], [Calendar.Year], 2, 'percentage')",
      "keywords": ["trend", "change", "growth", "yoy", "mom", "previous", "next", "compare", "lag", "lead", "%", "percentage"],
      "description": "Calculate trends and comparisons across time periods",
      "use_case": "Use for year-over-year, month-over-month, or any period comparisons",
      "options": ["average", "value", "percentage", "sum", "delta"],
      "notes": "trend_type options: 'average'=avg of previous periods, 'value'=value of previous period, 'percentage'=% change, 'sum'=sum of previous periods, 'delta'=change in value"
    },
    "PERIODSTODATE": {
      "syntax": "PERIODSTODATE(time_dimension_level, measure, period_type)",
      "example": "PERIODSTODATE([Calendar.FiYear], [Measures.PROFIT], 'avg')",
      "keywords": ["ytd", "mtd", "qtd", "year to date", "month to date", "quarter to date", "cumulative"],
      "description": "Calculate Year-to-Date, Month-to-Date, Quarter-to-Date values",
      "use_case": "Use when query asks for cumulative values up to current period",
      "options": ["avg", "sum"],
      "notes": "period_type: 'sum' for total accumulation, 'avg' for average accumulation"
    },
    "EOP": {
      "syntax": "EOP(kpi, time_dimension_level)",
      "example": "EOP([Measures.# of Employees], [Calendar.Year])",
      "keywords": ["end of period", "eop", "closing", "final", "last day", "period end"],
      "description": "Calculate End of Period values (end of year, month, etc.)",
      "use_case": "Use when you need the final value at the end of a time period"
    }
  },
  "ranking_functions": {
    "Head": {
      "syntax": "Head(dimension, measure, count, undefined)",
      "example": "Head([Branch Details].[City], [Business Drivers].[Balance Amount], 5, undefined)",
      "keywords": ["top", "best", "highest", "first", "maximum", "max", "greatest", "largest", "leading"],
      "description": "Get top N results based on a measure",
      "use_case": "Use when query asks for 'top X', 'best X', 'highest X' items"
    },
    "Tail": {
      "syntax": "Tail(dimension, measure, count, undefined)",
      "example": "Tail([Time].[Year], [Financial Data].[Total Revenue], 4, undefined)",
      "keywords": ["bottom", "worst", "lowest", "last", "minimum", "min", "smallest", "least"],
      "description": "Get bottom N results based on a measure",
      "use_case": "Use when query asks for 'bottom X', 'worst X', 'lowest X' items"
    },
    "rank": {
      "syntax": "rank(dimension_level, 'rankAcrossRows')",
      "example": "rank([Branch Details].[City], 'rankAcrossRows')",
      "keywords": ["rank", "ranking", "position", "order", "sequence"],
      "description": "Provide ranking numbers for values",
      "use_case": "Use when query asks for rankings or positions of items"
    }
  },
  "conditional_functions": {
    "IF": {
      "syntax": "IF(condition, TRUE_statement, FALSE_statement)",
      "example": "IF([Measures.Brokerage] = 0 or [Measures.Brokerage] = null, null, [Measures.Total Cost] / [Measures.Brokerage])",
      "keywords": ["if", "when", "condition", "then", "else", "case", "conditional"],
      "description": "Conditional IF-ELSE calculations on measures",
      "use_case": "Use when calculations depend on conditions or when handling null/zero values"
    },
    "ISPRESENT": {
      "syntax": "ISPRESENT(Level_Name)",
      "example": "IF(ISPRESENT([Branch.Zone]), [Measures.PROFIT], 0)",
      "keywords": ["present", "exists", "available", "has", "contains"],
      "description": "Check whether a level is present in the report",
      "use_case": "Use as condition in IF functions to check if a dimension level exists in the current context"
    },
    "FILTERKPI": {
      "syntax": "FILTERKPI(kpi, condition)",
      "example": "FILTERKPI([Measures.PROFIT], [Branch Type.Status] = 'ACTIVE')",
      "keywords": ["filter", "where", "condition", "only", "exclude", "conditional sum", "conditional average"],
      "description": "Conditional aggregation of measures with filters",
      "use_case": "Use for conditional sum, conditional average, or filtered aggregations"
    }
  },
  "aggregation_functions": {
    "SUM": {
      "syntax": "SUM(dimension_level, scope, kpi, dimension_level)",
      "example": "SUM([Customer.Customer Code], 'specific', [Measures.Exposure], [Account.Account Number])",
      "keywords": ["sum", "total", "aggregate", "add"],
      "description": "Custom aggregation at specific levels",
      "use_case": "Use for custom sum calculations at particular dimension levels",
      "scopes": ["specific", "current", "all"],
      "notes": "'specific'=aggregate at specified level, 'current'=current context, 'all'=all levels"
    },
    "MIN": {
      "syntax": "MIN(dimension_level, scope, kpi, dimension_level)",
      "example": "MIN([Customer.Customer Code], 'specific', [Measures.Balance], [Account.Account Number])",
      "keywords": ["min", "minimum", "smallest", "least"],
      "description": "Find minimum values at specific levels",
      "use_case": "Use to find minimum values across dimensions",
      "scopes": ["specific", "current", "all"]
    },
    "MAX": {
      "syntax": "MAX(dimension_level, scope, kpi, dimension_level)",
      "example": "MAX([Customer.Customer Code], 'specific', [Measures.Balance], [Account.Account Number])",
      "keywords": ["max", "maximum", "largest", "greatest"],
      "description": "Find maximum values at specific levels",
      "use_case": "Use to find maximum values across dimensions",
      "scopes": ["specific", "current", "all"]
    },
    "COUNT": {
      "syntax": "COUNT(dimension_level, scope, kpi, dimension_level)",
      "example": "COUNT([Customer.Customer Code], 'specific', [Measures.Transactions], [Account.Account Number])",
      "keywords": ["count", "number", "quantity", "total number"],
      "description": "Count items at specific levels",
      "use_case": "Use to count occurrences or items",
      "scopes": ["specific", "current", "all"]
    },
    "AVG": {
      "syntax": "AVG(dimension_level, scope, kpi, dimension_level)",
      "example": "AVG([Customer.Customer Code], 'specific', [Measures.Balance], [Account.Account Number])",
      "keywords": ["avg", "average", "mean"],
      "description": "Calculate average values at specific levels",
      "use_case": "Use to calculate averages across dimensions",
      "scopes": ["specific", "current", "all"]
    },
    "percentage": {
      "syntax": "percentage(measure, 'percentColumn')",
      "example": "percentage([Business Drivers].[Balance Amount], 'percentColumn')",
      "keywords": ["percentage", "percent", "%", "proportion", "ratio"],
      "description": "Calculate percentage values",
      "use_case": "Use when query asks for percentages or proportions"
    },
    "runningsum": {
      "syntax": "runningsum(measure, 'sumacrossrows')",
      "example": "runningsum([Business Drivers].[Balance Amount], 'sumacrossrows')",
      "keywords": ["running sum", "cumulative", "accumulative", "progressive total"],
      "description": "Calculate cumulative running totals",
      "use_case": "Use for running totals or cumulative calculations"
    },
    "percentageofrunningsum": {
      "syntax": "percentageofrunningsum(measure, 'percentagerunningsumacrossrows')",
      "example": "percentageofrunningsum([Business Drivers].[Balance Amount], 'percentagerunningsumacrossrows')",
      "keywords": ["percentage of running sum", "cumulative percentage", "running percentage"],
      "description": "Calculate percentage of running sum",
      "use_case": "Use for cumulative percentage calculations"
    }
  },
  "utility_functions": {
    "ROUND": {
      "syntax": "ROUND(kpi, decimal_places)",
      "example": "ROUND([Measures.PROFIT], 3)",
      "keywords": ["round", "decimal", "precision"],
      "description": "Round numbers to specified decimal places",
      "use_case": "Use to format numerical results with specific precision"
    },
    "COALESCE": {
      "syntax": "COALESCE(kpi, default_value)",
      "example": "COALESCE([Measures.PROFIT], 0)",
      "keywords": ["coalesce", "null", "default", "fallback"],
      "description": "Handle null values by providing default values",
      "use_case": "Use to replace null values with meaningful defaults"
    }
  },
  "comparison_functions": {
    "between": {
      "syntax": "measure between value1 and value2",
      "example": "[Business Drivers].[Balance Amount Average] between 400000000.00 and 2000000000.00",
      "keywords": ["between", "range", "within"],
      "description": "Filter values within a specific range",
      "use_case": "Use when filtering numerical values within min-max ranges"
    },
    "not_between": {
      "syntax": "measure not between value1 and value2", 
      "example": "[Business Drivers].[Balance Amount Average] not between 400000000.00 and 2000000000.00",
      "keywords": ["not between", "outside range", "excluding"],
      "description": "Filter values outside a specific range",
      "use_case": "Use when excluding values within a range"
    },
    "in": {
      "syntax": "dimension in ('value1','value2','value3')",
      "example": "[Mutual Fund Investment].[Mutual Fund Name] in ('AXIS','HDFC','ICICI','LIC')",
      "keywords": ["in", "among", "one of"],
      "description": "Filter for specific values from a list",
      "use_case": "Use when filtering for specific items from a predefined list"
    },
    "like": {
      "syntax": "dimension like 'pattern'",
      "example": "[Benchmark Index Details].[Index Name] like '%Nifty%'",
      "keywords": ["like", "contains", "similar", "pattern"],
      "description": "Pattern matching for text values",
      "use_case": "Use for text pattern matching or partial string searches"
    },
    "not_like": {
      "syntax": "dimension not like 'pattern'",
      "example": "[Benchmark Index Details].[Index Name] not like '%Nifty%'",
      "keywords": ["not like", "does not contain", "excluding pattern"],
      "description": "Exclude values matching a pattern",
      "use_case": "Use to exclude items that match a text pattern"
    }
  },
  "mathematical_operations": {
    "greater_than": {
      "syntax": "measure > value",
      "example": "[Business Drivers].[Balance Amount] > 0.00",
      "keywords": ["greater than", "more than", "above", "exceeds"],
      "description": "Filter values greater than a threshold",
      "use_case": "Use when filtering for values above a certain limit"
    },
    "less_than": {
      "syntax": "measure < value",
      "example": "[Bulk Deal Trade].[Trade Price] < 276.00",
      "keywords": ["less than", "below", "under", "smaller"],
      "description": "Filter values less than a threshold", 
      "use_case": "Use when filtering for values below a certain limit"
    },
    "greater_equal": {
      "syntax": "measure >= value",
      "example": "[Business Drivers].[Count of Customers] >= 10.00",
      "keywords": ["greater than or equal", "at least", "minimum"],
      "description": "Filter values greater than or equal to a threshold",
      "use_case": "Use when setting minimum value requirements"
    },
    "less_equal": {
      "syntax": "measure <= value", 
      "example": "[Business Drivers].[Balance Amount] <= 1000000.00",
      "keywords": ["less than or equal", "at most", "maximum"],
      "description": "Filter values less than or equal to a threshold",
      "use_case": "Use when setting maximum value limits"
    },
    "equals": {
      "syntax": "dimension = 'value' or measure = value",
      "example": "[Customer Details].[EWS Tag] = 'HIGH' and [Time].[Day] = '2025-01-15'",
      "keywords": ["equals", "is", "equal to"],
      "description": "Filter for exact value matches",
      "use_case": "Use when filtering for exact values"
    }
  },
  "logical_operators": {
    "and": {
      "syntax": "condition1 and condition2",
      "example": "[Business Drivers].[Count of Customers] > 10.00 and [Business Drivers].[Balance Amount Average] > 5000.00",
      "keywords": ["and", "also", "both", "additionally"],
      "description": "Combine multiple conditions (all must be true)",
      "use_case": "Use when all conditions must be satisfied"
    },
    "or": {
      "syntax": "condition1 or condition2", 
      "example": "[Measures.Brokerage] = 0 or [Measures.Brokerage] = null",
      "keywords": ["or", "either", "alternatively"],
      "description": "Combine multiple conditions (any can be true)",
      "use_case": "Use when any of the conditions can be satisfied"
    }
  }
}
