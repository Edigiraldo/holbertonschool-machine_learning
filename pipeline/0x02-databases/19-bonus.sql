-- script that creates a stored procedure AddBonus that adds a new correction for a student.
DELIMITER $$ ;
CREATE PROCEDURE AddBonus (IN user_id int, IN project_name VARCHAR(255), IN score int)
BEGIN
  DECLARE project_id int DEFAULT -1;
  IF NOT EXISTS (SELECT * FROM projects WHERE projects.name = project_name) THEN
    INSERT INTO projects (name) VALUES (project_name);
  END IF;
  SET project_id = (SELECT id FROM projects WHERE name = project_name);
  INSERT INTO corrections VALUES (user_id, project_id, score);
END$$
DELIMITER ; $$
