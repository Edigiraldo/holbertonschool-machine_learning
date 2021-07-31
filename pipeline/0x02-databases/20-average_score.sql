-- script that creates a stored procedure ComputeAverageScoreForUser that computes and store the average score for a student.
DELIMITER $$ ;
CREATE PROCEDURE ComputeAverageScoreForUser (IN u_id int)
BEGIN
  DECLARE avg_score int;
  SET avg_score = (SELECT AVG(score) FROM corrections WHERE user_id = u_id);
  UPDATE users SET average_score = avg_score WHERE id = u_id;
END$$
DELIMITER ; $$
