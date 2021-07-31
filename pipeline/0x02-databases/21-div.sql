-- script that creates a function SafeDiv.
DELIMITER $$ ;
CREATE FUNCTION SafeDiv (a INT, b INT) RETURNS FLOAT DETERMINISTIC
BEGIN
  IF (b = 0) THEN
    SET @result = 0;
  ELSE
    SET @result =  a / b;
  END IF;
  RETURN @result ;
END $$
DELIMITER ; $$
