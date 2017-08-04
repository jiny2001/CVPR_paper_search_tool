package models;

import java.util.List;

import javax.persistence.*;

import play.db.ebean.*;


/**
 * Company entity managed by Ebean
 */
@Entity 
public class Paper extends Model {

    private static final long serialVersionUID = 1L;

	@Id
    public Long id;
    
    public String title;
    public String abstract_text;
    public String abstract_url;
    public String pdf_url;
    public String pdf_text;
    
    public static Model.Finder<Long,Paper> find = new Model.Finder<Long,Paper>(Long.class, Paper.class);

    public static List<Paper> list() {

        List<Paper> papers = find.all();

        return papers;
    }
}

