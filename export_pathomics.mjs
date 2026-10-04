/** Native Excel exporter. Needs Node + @oai/artifact-tool in this environment.
 * Server extraction has NO Node requirement: transfer workbook_data.json here.
 * Usage: node export_pathomics.mjs workbook_data.json [output.xlsx] [--preview]
 */
import fs from 'node:fs/promises';
import path from 'node:path';
import {Workbook, SpreadsheetFile} from '@oai/artifact-tool';

const positional=process.argv.slice(2).filter(x=>x!=='--preview');
if(!positional[0]) throw Error('Usage: node export_pathomics.mjs workbook_data.json [output.xlsx] [--preview]');
const src=path.resolve(positional[0]);
const output=positional[1] ? path.resolve(positional[1]) : path.join(path.dirname(src),'생검_WSI_정량특징.xlsx');
const data=JSON.parse(await fs.readFile(src,'utf8'));
const wb=Workbook.create();
function letter(n){let s='';for(let x=n;x;x=Math.floor((x-1)/26))s=String.fromCharCode(65+(x-1)%26)+s;return s;}
function sheet(name,rows,headers){
 const s=wb.worksheets.add(name);s.showGridLines=false;
 const cols=headers??[...new Set(rows.flatMap(r=>Object.keys(r)))];
 if(!cols.length)cols.push('status');
 if(cols.length>16384||rows.length>1048575)throw Error(`Excel limits exceeded: ${name}; use CSV`);
 const end=letter(cols.length);
 s.getRange(`A1:${end}1`).values=[cols];
 // Imported measurements are constants; identity columns remain text. Empty
 // string is not a zero measurement. Formula-like identifiers are escaped.
 const value=v=>v===null||v===undefined?null:typeof v==='object'?JSON.stringify(v):typeof v==='string'&&/^[=+@]/.test(v)?`'${v}`:v;
 for(let i=0;i<rows.length;i+=100){
   const chunk=rows.slice(i,i+100);
   s.getRange(`A${i+2}:${end}${i+chunk.length+1}`).values=chunk.map(r=>cols.map(k=>value(r[k])));
 }
 s.getRange(`A1:${end}1`).format={fill:'#18394a',font:{bold:true,color:'#FFFFFF',size:10},wrapText:true,rowHeight:78,verticalAlignment:'center'};
 s.getRange(`A:${end}`).format.columnWidth=21;
 s.getRange('A:D').format.columnWidth=27;
 if(rows.length){
   const table=s.tables.add(`A1:${end}${rows.length+1}`,true,`Table_${name}`);table.showBandedRows=true;
   s.getRange(`A2:${end}${rows.length+1}`).format.rowHeight=21;
   for(let i=0;i<cols.length;i++){
     const k=cols[i], col=letter(i+1),sample=rows.find(r=>r[k]!==null&&r[k]!==undefined)?.[k];
     if(typeof sample==='number'&&!/count|valid_n|patches|slides|eligible|completed|nodes|edges|index|excluded|core_report/.test(k))s.getRange(`${col}2:${col}${rows.length+1}`).setNumberFormat('0.0000');
     if(/sha256|signature/.test(k)){
       s.getRange(`${col}:${col}`).format.columnWidth=35;
       s.getRange(`${col}2:${col}${rows.length+1}`).format.wrapText=true;
       s.getRange(`A2:${end}${rows.length+1}`).format.rowHeight=50;
     }
   }
 }
 s.freezePanes.freezeRows(1);s.freezePanes.freezeColumns(name==='Biopsy_Features'?4:1);
 return s;
}
sheet('Biopsy_Features',data.main);
sheet('Slide_QC',data.slides);
const dict=sheet('Feature_Dictionary',data.dictionary);
dict.getRange('A:A').format.columnWidth=49;dict.getRange('D:H').format.columnWidth=47;
dict.getRange('F:F').format.columnWidth=14;
dict.getRange(`A2:H${data.dictionary.length+1}`).format={wrapText:true,rowHeight:60,verticalAlignment:'center'};
sheet('Exclusions',data.exclusions);
sheet('Extraction_Audit',data.receipts);
const notes=[['Run mode',data.smoke_only?'SYNTHETIC/SMOKE ONLY — NOT clinical results':'Full selected original H&E cores'],...data.notes];
const methods=sheet('Methods',notes.map(([item,description])=>({item,description})));
methods.getRange('A:A').format.columnWidth=24;methods.getRange('B:B').format.columnWidth=110;
methods.getRange(`A2:B${notes.length+1}`).format={wrapText:true,rowHeight:65,verticalAlignment:'center'};
const scan=await wb.inspect({kind:'match',searchTerm:'#REF!|#DIV/0!|#VALUE!|#NAME\\?|#NUM!',options:{useRegex:true,maxResults:20}});
console.log(scan.ndjson);
if(process.argv.includes('--preview')){
 for(const s of wb.worksheets.items){
  const range=s.name==='Methods'?'A1:B9':s.name==='Feature_Dictionary'?'A1:G7':'A1:J8';
  const blob=await wb.render({sheetName:s.name,range,scale:1,format:'png'});
  await fs.writeFile(path.join(path.dirname(output),`preview_${s.name}.png`),new Uint8Array(await blob.arrayBuffer()));
 }
 const features=await wb.render({sheetName:'Biopsy_Features',range:'M1:V5',scale:1,format:'png'});
 await fs.writeFile(path.join(path.dirname(output),'preview_FeatureValues.png'),new Uint8Array(await features.arrayBuffer()));
}
await (await SpreadsheetFile.exportXlsx(wb)).save(output);
console.log(`Saved ${output}; ${data.main.length} biopsy rows. No model was trained.`);
