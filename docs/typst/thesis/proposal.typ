#import "template/layout/proposal_template.typ": *
#import "metadata.typ": *
#import "../shared/macros.typ": *
#import "../shared/glossary.typ": *

#set document(title: titleEnglish + " Proposal", author: author)
#set text(font: "New Computer Modern")

#show: proposal.with(
  title: titleEnglish,
  titleGerman: titleGerman,
  thesisKindEnglish: thesisKindEnglish + " Proposal",
  thesisKindGerman: "Proposal zur " + thesisKindGerman,
  academicDegree: academicDegree,
  program: program,
  specialization: specialization,
  universityEnglish: universityEnglish,
  universityGerman: universityGerman,
  facultyEnglish: facultyEnglish,
  facultyGerman: facultyGerman,
  firstExaminer: firstExaminer,
  secondExaminer: secondExaminer,
  supervisors: supervisors,
  author: author,
  email: email,
  matriculationNumber: matriculationNumber,
  startDate: startDate,
  submissionDate: submissionDate,
  submissionDateText: submissionDateText,
  transparency_ai_tools: [
    TODO: Document AI tools used while preparing the proposal according to the final institutional requirements.
  ],
)

#include "sections/proposal/01-motivation.typ"
#include "sections/proposal/02-problem.typ"
#include "sections/proposal/03-objectives.typ"
#include "sections/proposal/04-method.typ"
#include "sections/proposal/05-schedule.typ"
#include "sections/proposal/06-outline.typ"
