-- Q1: Customers whose completed order spending exceeds INR 8000
SELECT
    customer_id,
    SUM(order_amount) AS total_spend
FROM orders
WHERE status = 'completed'
    AND order_amount > 500
GROUP BY customer_id
HAVING SUM(order_amount) > 8000;

-- Q2: Revenue by city and product category
SELECT
    c.city,
    p.category,
    SUM(o.order_amount) AS revenue
FROM orders o
JOIN customers c
    ON o.customer_id = c.customer_id
JOIN products p
    ON o.product_id = p.product_id
WHERE o.status = 'completed'
GROUP BY c.city, p.category
ORDER BY revenue DESC;

-- Q3: Products priced above average catalogue price
SELECT *
FROM products
WHERE price_inr > (
    SELECT AVG(price_inr)
    FROM products
);

-- Q4: Customers with no completed orders
SELECT
    c.customer_id,
    c.city
FROM customers c
LEFT JOIN orders o
    ON c.customer_id = o.customer_id
    AND o.status = 'completed'
WHERE o.customer_id IS NULL;

-- Q5: Customers spending above average
WITH customer_spend AS (
    SELECT
        customer_id,
        SUM(order_amount) AS total_spend
    FROM orders
    WHERE status = 'completed'
    GROUP BY customer_id
),
average_spend AS (
    SELECT AVG(total_spend) AS avg_spend
    FROM customer_spend
)
SELECT *
FROM customer_spend
WHERE total_spend > (
    SELECT avg_spend
    FROM average_spend
);

-- Q6: Customer spending rankings within each city
WITH customer_spend AS (
    SELECT
        c.customer_id,
        c.city,
        SUM(o.order_amount) AS total_spend
    FROM customers c
    JOIN orders o
        ON c.customer_id = o.customer_id
    WHERE o.status = 'completed'
    GROUP BY c.customer_id, c.city
)
SELECT
    customer_id,
    city,
    total_spend,
    RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS rank_num,
    DENSE_RANK() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS dense_rank_num,
    ROW_NUMBER() OVER (
        PARTITION BY city
        ORDER BY total_spend DESC
    ) AS row_num
FROM customer_spend
ORDER BY city, total_spend DESC;

-- Q7: Daily revenue and cumulative running total
WITH daily_revenue AS (
    SELECT
        DATE(order_ts) AS day,
        SUM(order_amount) AS revenue
    FROM orders
    WHERE status = 'completed'
    GROUP BY DATE(order_ts)
)
SELECT
    day,
    revenue,
    SUM(revenue) OVER (
        ORDER BY day
    ) AS running_total
FROM daily_revenue
ORDER BY day;