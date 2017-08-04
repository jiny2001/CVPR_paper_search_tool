# --- First database schema

# --- !Ups

create table paper (
  id                        bigint not null,
  title                      varchar(1024),
  abstract_text                      text,
  abstract_url                      varchar(1024),
  pdf_url                      varchar(1024),
  pdf_text                      mediumtext,
    
  constraint pk_paper primary key (id))
;
# --- !Downs

drop table if exists paper;


